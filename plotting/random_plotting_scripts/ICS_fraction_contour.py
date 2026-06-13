#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob

from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings("ignore")


def _int_levels(H, n=15):
    """Integer contour levels from 1 to max(H) for halo-count histograms."""
    if not np.any(~np.isnan(H)):
        return np.array([1, 2], dtype=float)
    mx = int(np.nanmax(H))
    if mx < 2:
        return np.array([1, 2], dtype=float)
    if mx <= n:
        return np.arange(1, mx + 1, dtype=float)
    return np.unique(np.round(np.linspace(1, mx, n)).astype(int)).astype(float)


def _make_transparent_cmap(base_cmap_name):
    """Create a colormap that fades from fully transparent to the base colormap."""
    base = plt.cm.get_cmap(base_cmap_name, 256)
    colors = base(np.linspace(0, 1, 256))
    colors[:, 3] = np.linspace(0.7, 1, 256)  # alpha: 0.7 (semi-transparent) -> 1 (opaque)
    return LinearSegmentedColormap.from_list(base_cmap_name + '_t', colors)

# ========================== USER OPTIONS ==========================

OutputFormat = '.pdf'

plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.frameon'] = False

# Try to load project style
_style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(_style_path):
    plt.style.use(_style_path)

# ==================================================================


def read_simulation_params(filepath):
    """Read simulation parameters from HDF5 file header."""
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        # particle_mass is in 1e10 Msun/h units; convert to Msun
        params['PartMass'] = float(sim.attrs['particle_mass']) * 1.0e10 / float(sim.attrs['hubble_h'])

        runtime = f['Header/Runtime']
        params['BaryonFrac'] = float(runtime.attrs.get('BaryonFrac', 0.17))

        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['last_snapshot'] = max(snap_numbers) if snap_numbers else 0

    return params


def read_snapshot_data(file_list, snap_name, fields, hubble_h):
    """Read and concatenate fields from all HDF5 files for a given snapshot."""
    mass_fields = {'Mvir', 'IntraClusterStars', 'StellarMass', 'ColdGas',
                   'HotGas', 'CGMgas', 'EjectedMass', 'BlackHoleMass',
                   'MetalsIntraClusterStars', 'MetalsStellarMass',
                   'MetalsBulgeMass', 'MetalsColdGas', 'MetalsHotGas',
                   'MetalsCGMgas', 'MetalsEjectedMass'}

    combined = {f: [] for f in fields}
    for filepath in file_list:
        with h5.File(filepath, 'r') as f:
            if snap_name not in f:
                continue
            grp = f[snap_name]
            for field in fields:
                if field in grp:
                    data = np.array(grp[field])
                    if data.size > 0:
                        if field in mass_fields:
                            data = data * 1.0e10 / hubble_h
                        combined[field].append(data)

    result = {}
    for field in fields:
        if combined[field]:
            result[field] = np.concatenate(combined[field])
        else:
            result[field] = np.array([])
    return result


def load_ics_data(file_list, sim_params, snap_num):
    """Load snapshot data and compute satellite counts and stellar mass per central."""
    hubble_h = sim_params['Hubble_h']
    snap_name = f'Snap_{snap_num}'
    redshift = sim_params['redshifts'][snap_num]

    data = read_snapshot_data(file_list, snap_name,
                              ['Mvir', 'IntraClusterStars', 'MetalsIntraClusterStars',
                               'StellarMass', 'MetalsStellarMass',
                               'ColdGas', 'MetalsColdGas', 'Concentration',
                               'Type', 'GalaxyIndex', 'CentralGalaxyIndex'],
                              hubble_h)

    if data['Mvir'].size == 0:
        return None

    Mvir = data['Mvir']
    ICS = data['IntraClusterStars']
    MetalsICS = data['MetalsIntraClusterStars']
    StellarMass = data['StellarMass']
    MetalsStellarMass = data['MetalsStellarMass']
    ColdGas = data['ColdGas']
    MetalsColdGas = data['MetalsColdGas']
    Concentration = data['Concentration']
    Type = data['Type']
    GalaxyIndex = data['GalaxyIndex']
    CentralGalaxyIndex = data['CentralGalaxyIndex']

    # Count satellites and sum satellite stellar mass per central
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]

    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    sat_stellar = StellarMass[sat_mask]

    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_sats = central_indices >= 0

    n_satellites = np.zeros(len(Type), dtype=int)
    np.add.at(n_satellites, central_indices[valid_sats], 1)

    halo_stellar_mass = np.copy(StellarMass)
    np.add.at(halo_stellar_mass, central_indices[valid_sats], sat_stellar[valid_sats])

    return {
        'Mvir': Mvir, 'ICS': ICS, 'MetalsICS': MetalsICS,
        'StellarMass': StellarMass, 'MetalsStellarMass': MetalsStellarMass,
        'ColdGas': ColdGas, 'MetalsColdGas': MetalsColdGas,
        'Concentration': Concentration,
        'HaloStellarMass': halo_stellar_mass,
        'Type': Type, 'n_satellites': n_satellites, 'redshift': redshift,
    }


def plot_ics_fraction_contour(file_list, sim_params, snap_num, output_dir):
    """
    Contour plot of ICS fraction (f_ICS = M_ICS / (f_b * Mvir))
    as a function of halo mass, using 2D histogram density.
    Includes 3 panels: Groups/Clusters, Isolated, and Combined.
    """
    hubble_h = sim_params['Hubble_h']
    baryon_frac = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * baryon_frac
    snap_name = f'Snap_{snap_num}'
    redshift = sim_params['redshifts'][snap_num]

    print(f'Reading snapshot {snap_num} (z = {redshift:.2f})...')

    data = read_snapshot_data(file_list, snap_name,
                              ['Mvir', 'IntraClusterStars', 'StellarMass',
                               'Type', 'GalaxyIndex', 'CentralGalaxyIndex'],
                              hubble_h)

    Mvir = data['Mvir']
    ICS = data['IntraClusterStars']
    StellarMass = data['StellarMass']
    Type = data['Type']
    GalaxyIndex = data['GalaxyIndex']
    CentralGalaxyIndex = data['CentralGalaxyIndex']

    if Mvir.size == 0:
        print(f'  No data found in {snap_name}')
        return

    # Count satellites per central using searchsorted
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]

    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]

    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)

    n_satellites = np.zeros(len(Type), dtype=int)
    np.add.at(n_satellites, central_indices[central_indices >= 0], 1)

    # Split into groups/clusters (>= 1 satellite) and isolated (0 satellites)
    base_mask = (Type == 0) & (ICS > 0) & (Mvir > 0) & (StellarMass >= min_stellar)
    gc_mask = base_mask & (n_satellites >= 1)
    iso_mask = base_mask & (n_satellites == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals with ICS > 0 (>= 1 satellite)')
    print(f'  {np.sum(iso_mask)} isolated centrals with ICS > 0 (0 satellites)')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    # Compute ICS fractions for both populations
    log_Mvir_gc = np.log10(Mvir[gc_mask])
    log_fICS_gc = np.log10(ICS[gc_mask] / (baryon_frac * Mvir[gc_mask]))
    n_sat_gc = n_satellites[gc_mask]

    log_Mvir_iso = np.log10(Mvir[iso_mask])
    log_fICS_iso = np.log10(ICS[iso_mask] / (baryon_frac * Mvir[iso_mask]))

    # Common axis ranges across panels
    all_log_Mvir = np.concatenate([log_Mvir_gc, log_Mvir_iso])
    all_log_fICS = np.concatenate([log_fICS_gc, log_fICS_iso])
    x_lo, x_hi = all_log_Mvir.min() - 0.1, all_log_Mvir.max() + 0.1
    y_lo, y_hi = all_log_fICS.min() - 0.25, all_log_fICS.max() + 0.5

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    # Updated to 3 subplots and widened the figure
    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True,
                                                 figsize=(24, 6.25))

    mass_bins = np.arange(11.0, 16.1, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mvir_gc, log_fICS_gc,
                                           bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)

    H_gc = H_gc.T
    H_gc_masked = np.where(H_gc > 0, H_gc, np.nan)

    cf_gc = ax_gc.contourf(Xg, Yg, H_gc_masked, levels=_int_levels(H_gc_masked), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mvir_gc >= mass_bins[i]) & (log_Mvir_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_fICS_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
            
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
        print(f'  Groups/Clusters: {np.sum(gc_mask)} haloes')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_ylabel(r'$\log_{10}\, f_{\mathrm{ICS}}\ =\ m_{\mathrm{ICS}} / (f_b\ M_{\mathrm{vir}})$')

    # --- Middle panel: Isolated haloes ---
    H_iso, xedges, yedges = np.histogram2d(log_Mvir_iso, log_fICS_iso,
                                            bins=[x_bins, y_bins])
    H_iso = H_iso.T
    H_iso_masked = np.where(H_iso > 0, H_iso, np.nan)

    cf_iso = ax_iso.contourf(Xg, Yg, H_iso_masked, levels=_int_levels(H_iso_masked), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mvir_iso >= mass_bins[i]) & (log_Mvir_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_fICS_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
            
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    
    # --- Right panel: Combined population ---
    H_comb, _, _ = np.histogram2d(all_log_Mvir, all_log_fICS, 
                                  bins=[x_bins, y_bins])
    H_comb = H_comb.T
    H_comb_masked = np.where(H_comb > 0, H_comb, np.nan)

    cf_comb = ax_comb.contourf(Xg, Yg, H_comb_masked, levels=_int_levels(H_comb_masked), cmap='RdPu_r')

    # Plot both medians on the combined panel
    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label=r'Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
        
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')

    # --- Colorbars ---
    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    
    # Assuming OutputFormat is defined globally or passed in, default to .png if missing
    OutputFormat = globals().get('OutputFormat', '.png')
    outfile = os.path.join(output_dir,
                           f'ICS_fraction_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_stellar_fraction_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of f_ICS = M_ICS / (M_*,halo + M_ICS) vs halo mass.
    Left: Groups/clusters, Middle: Isolated, Right: Combined population with both medians.
    """
    redshift = sim_params['redshifts'][snap_num]
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nStellar-fraction 3-panel contour: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = (d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0) & (d['StellarMass'] >= min_stellar)
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies.')
        return

    ylabel = r'$\log_{10}\, f_{\mathrm{ICS}}\ =\ M_{\mathrm{ICS}} / (M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$'

    # Compute f_ICS for both populations
    log_Mvir_gc = np.log10(d['Mvir'][gc_mask])
    log_fICS_gc = np.log10(d['ICS'][gc_mask] / (d['HaloStellarMass'][gc_mask] + d['ICS'][gc_mask]))

    log_Mvir_iso = np.log10(d['Mvir'][iso_mask])
    log_fICS_iso = np.log10(d['ICS'][iso_mask] / (d['HaloStellarMass'][iso_mask] + d['ICS'][iso_mask]))

    # Common axis ranges
    all_log_Mvir = np.concatenate([log_Mvir_gc, log_Mvir_iso])
    all_log_fICS = np.concatenate([log_fICS_gc, log_fICS_iso])
    x_lo, x_hi = all_log_Mvir.min() - 0.1, all_log_Mvir.max() + 0.1
    y_lo, y_hi = all_log_fICS.min() - 0.25, all_log_fICS.max() + 0.5

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    # Updated to 3 subplots and widened figure
    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(11.0, 16.1, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mvir_gc, log_fICS_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)

    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mvir_gc >= mass_bins[i]) & (log_Mvir_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_fICS_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
            
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mvir_iso, log_fICS_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mvir_iso >= mass_bins[i]) & (log_Mvir_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_fICS_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
            
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')

    # --- Right panel: Combined population ---
    H_comb, _, _ = np.histogram2d(all_log_Mvir, all_log_fICS, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    # Plot both medians on the combined panel
    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label=r'Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
        
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')

    # --- Colorbars ---
    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    
    # Resolving output format securely
    OutputFormat = globals().get('OutputFormat', '.png')
    outfile = os.path.join(output_dir,
                           f'ICS_stellar_fraction_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_vs_bcg_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log M_ICS vs log M_BCG (central stellar mass).
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS vs BCG 3-panel contour: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = (d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0) & (d['StellarMass'] >= min_stellar)
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Mbcg_gc = np.log10(d['StellarMass'][gc_mask])
    log_Mics_gc = np.log10(d['ICS'][gc_mask])
    log_Mbcg_iso = np.log10(d['StellarMass'][iso_mask])
    log_Mics_iso = np.log10(d['ICS'][iso_mask])

    all_log_Mbcg = np.concatenate([log_Mbcg_gc, log_Mbcg_iso])
    all_log_Mics = np.concatenate([log_Mics_gc, log_Mics_iso])

    x_lo, x_hi = all_log_Mbcg.min() - 0.1, all_log_Mbcg.max() + 0.1
    y_lo, y_hi = all_log_Mics.min() - 0.25, all_log_Mics.max() + 0.5

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mbcg_gc, log_Mics_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_gc >= mass_bins[i]) & (log_Mbcg_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Mics_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(r'$\log_{10}\, M_{\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_ylabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$')

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mbcg_iso, log_Mics_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_iso >= mass_bins[i]) & (log_Mbcg_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Mics_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(r'$\log_{10}\, M_{\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mbcg, all_log_Mics, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(r'$\log_{10}\, M_{\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_vs_BCG_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log M_ICS vs log M_*,halo (total stellar mass in halo).
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS vs Halo Stellar Mass 3-panel contour: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = (d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0) & (d['StellarMass'] >= min_stellar)
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Mstar_gc = np.log10(d['HaloStellarMass'][gc_mask])
    log_Mics_gc = np.log10(d['ICS'][gc_mask])
    log_Mstar_iso = np.log10(d['HaloStellarMass'][iso_mask])
    log_Mics_iso = np.log10(d['ICS'][iso_mask])

    all_log_Mstar = np.concatenate([log_Mstar_gc, log_Mstar_iso])
    all_log_Mics = np.concatenate([log_Mics_gc, log_Mics_iso])

    x_lo, x_hi = all_log_Mstar.min() - 0.1, all_log_Mstar.max() + 0.1
    y_lo, y_hi = all_log_Mics.min() - 0.25, all_log_Mics.max() + 0.5

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mstar_gc, log_Mics_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_gc >= mass_bins[i]) & (log_Mstar_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Mics_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_ylabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$')

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mstar_iso, log_Mics_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_iso >= mass_bins[i]) & (log_Mstar_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Mics_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mstar, all_log_Mics, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_vs_HaloStellar_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log ICS metallicity (Z/Z_sun) vs log M_*,halo.
    Panels: Groups/Clusters, Isolated, Combined.
    """
    Z_SUN = 0.02
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity vs Halo Stellar Mass 3-panel contour: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['MetalsICS'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    # Metallicity Z = MetalsICS / ICS; then express in solar units
    log_Mstar_gc = np.log10(d['HaloStellarMass'][gc_mask])
    log_Z_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / Z_SUN)
    log_Mstar_iso = np.log10(d['HaloStellarMass'][iso_mask])
    log_Z_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / Z_SUN)

    all_log_Mstar = np.concatenate([log_Mstar_gc, log_Mstar_iso])
    all_log_Z = np.concatenate([log_Z_gc, log_Z_iso])

    x_lo, x_hi = all_log_Mstar.min() - 0.1, all_log_Mstar.max() + 0.1
    y_lo, y_hi = all_log_Z.min() - 0.25, all_log_Z.max() + 0.25

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mstar_gc, log_Z_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_gc >= mass_bins[i]) & (log_Mstar_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Z_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_ylabel(r'$\log_{10}\, (Z_{\mathrm{ICS}} / Z_{\odot})$')

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mstar_iso, log_Z_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_iso >= mass_bins[i]) & (log_Mstar_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Z_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mstar, all_log_Z, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$')

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity_vs_HaloStellar_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity12_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of ICS metallicity as log10(Z_ICS / 0.02) + 9.0 vs log M_*,halo.
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity (12+logZ/0.02) vs Halo Stellar Mass: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['MetalsICS'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Mstar_gc = np.log10(d['HaloStellarMass'][gc_mask])
    log_Z_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / 0.02) + 9.0
    log_Mstar_iso = np.log10(d['HaloStellarMass'][iso_mask])
    log_Z_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / 0.02) + 9.0

    all_log_Mstar = np.concatenate([log_Mstar_gc, log_Mstar_iso])
    all_log_Z = np.concatenate([log_Z_gc, log_Z_iso])

    x_lo, x_hi = all_log_Mstar.min() - 0.1, all_log_Mstar.max() + 0.1
    y_lo, y_hi = all_log_Z.min() - 0.25, all_log_Z.max() + 0.25

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    ylabel = r'$12 + \log_{10}\, (Z_{\mathrm{ICS}} / 0.02)$'
    xlabel = r'$\log_{10}\, M_{*,\mathrm{halo}}\ [\mathrm{M}_{\odot}]$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mstar_gc, log_Z_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_gc >= mass_bins[i]) & (log_Mstar_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Z_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mstar_iso, log_Z_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mstar_iso >= mass_bins[i]) & (log_Mstar_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Z_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mstar, all_log_Z, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity12_vs_HaloStellar_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity_vs_bcg_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log ICS metallicity (Z/Z_sun) vs log M_BCG (central stellar mass).
    Panels: Groups/Clusters, Isolated, Combined.
    """
    Z_SUN = 0.02
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity vs BCG Mass: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['MetalsICS'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Mbcg_gc = np.log10(d['StellarMass'][gc_mask])
    log_Z_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / Z_SUN)
    log_Mbcg_iso = np.log10(d['StellarMass'][iso_mask])
    log_Z_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / Z_SUN)

    all_log_Mbcg = np.concatenate([log_Mbcg_gc, log_Mbcg_iso])
    all_log_Z = np.concatenate([log_Z_gc, log_Z_iso])

    x_lo, x_hi = all_log_Mbcg.min() - 0.1, all_log_Mbcg.max() + 0.1
    y_lo, y_hi = all_log_Z.min() - 0.25, all_log_Z.max() + 0.25

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    ylabel = r'$\log_{10}\, (Z_{\mathrm{ICS}} / Z_{\odot})$'
    xlabel = r'$\log_{10}\, M_{\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mbcg_gc, log_Z_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_gc >= mass_bins[i]) & (log_Mbcg_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Z_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mbcg_iso, log_Z_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_iso >= mass_bins[i]) & (log_Mbcg_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Z_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mbcg, all_log_Z, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity_vs_BCG_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity12_vs_bcg_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of ICS metallicity as log10(Z_ICS / 0.02) + 9.0 vs log M_BCG.
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity (12+logZ/0.02) vs BCG Mass: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['MetalsICS'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Mbcg_gc = np.log10(d['StellarMass'][gc_mask])
    log_Z_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / 0.02) + 9.0
    log_Mbcg_iso = np.log10(d['StellarMass'][iso_mask])
    log_Z_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / 0.02) + 9.0

    all_log_Mbcg = np.concatenate([log_Mbcg_gc, log_Mbcg_iso])
    all_log_Z = np.concatenate([log_Z_gc, log_Z_iso])

    x_lo, x_hi = all_log_Mbcg.min() - 0.1, all_log_Mbcg.max() + 0.1
    y_lo, y_hi = all_log_Z.min() - 0.25, all_log_Z.max() + 0.25

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    mass_bins = np.arange(x_lo, x_hi + 0.2, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    ylabel = r'$12 + \log_{10}\, (Z_{\mathrm{ICS}} / 0.02)$'
    xlabel = r'$\log_{10}\, M_{\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Mbcg_gc, log_Z_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_gc >= mass_bins[i]) & (log_Mbcg_gc < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Z_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Mbcg_iso, log_Z_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(mass_bins) - 1):
        in_bin = (log_Mbcg_iso >= mass_bins[i]) & (log_Mbcg_iso < mass_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Z_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Mbcg, all_log_Z, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity12_vs_BCG_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity_vs_bcg_metallicity_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log(Z_ICS/Z_sun) vs log(Z_BCG/Z_sun).
    Panels: Groups/Clusters, Isolated, Combined.
    """
    Z_SUN = 0.02
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity vs BCG Metallicity: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0) & (d['ColdGas'] > 0) & (d['MetalsColdGas'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Zbcg_gc = np.log10(d['MetalsColdGas'][gc_mask] / d['ColdGas'][gc_mask] / Z_SUN)
    log_Zics_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / Z_SUN)
    log_Zbcg_iso = np.log10(d['MetalsColdGas'][iso_mask] / d['ColdGas'][iso_mask] / Z_SUN)
    log_Zics_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / Z_SUN)

    all_log_Zbcg = np.concatenate([log_Zbcg_gc, log_Zbcg_iso])
    all_log_Zics = np.concatenate([log_Zics_gc, log_Zics_iso])

    x_lo, x_hi = all_log_Zbcg.min() - 0.1, all_log_Zbcg.max() + 0.1
    y_lo, y_hi = all_log_Zics.min() - 0.1, all_log_Zics.max() + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    met_bins = np.arange(x_lo, x_hi + 0.1, 0.1)
    bin_centres = 0.5 * (met_bins[:-1] + met_bins[1:])

    ylabel = r'$\log_{10}\, (Z_{\mathrm{ICS}} / Z_{\odot})$'
    xlabel = r'$\log_{10}\, (Z_{\mathrm{BCG}} / Z_{\odot})$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Zbcg_gc, log_Zics_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(met_bins) - 1):
        in_bin = (log_Zbcg_gc >= met_bins[i]) & (log_Zbcg_gc < met_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Zics_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    # 1:1 line
    lo_line = min(x_lo, y_lo)
    hi_line = max(x_hi, y_hi)
    ax_gc.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Zbcg_iso, log_Zics_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(met_bins) - 1):
        in_bin = (log_Zbcg_iso >= met_bins[i]) & (log_Zbcg_iso < met_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Zics_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Zbcg, all_log_Zics, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity_vs_BCG_Metallicity_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_metallicity12_vs_bcg_metallicity12_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of 12+log(Z_ICS/0.02) vs 12+log(Z_BCG/0.02).
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS Metallicity (12+logZ) vs BCG Metallicity (12+logZ): snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0) & (d['ColdGas'] > 0) & (d['MetalsColdGas'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    log_Zbcg_gc = np.log10(d['MetalsColdGas'][gc_mask] / d['ColdGas'][gc_mask] / 0.02) + 9.0
    log_Zics_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / 0.02) + 9.0
    log_Zbcg_iso = np.log10(d['MetalsColdGas'][iso_mask] / d['ColdGas'][iso_mask] / 0.02) + 9.0
    log_Zics_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / 0.02) + 9.0

    all_log_Zbcg = np.concatenate([log_Zbcg_gc, log_Zbcg_iso])
    all_log_Zics = np.concatenate([log_Zics_gc, log_Zics_iso])

    x_lo, x_hi = all_log_Zbcg.min() - 0.1, all_log_Zbcg.max() + 0.1
    y_lo, y_hi = all_log_Zics.min() - 0.1, all_log_Zics.max() + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    met_bins = np.arange(x_lo, x_hi + 0.1, 0.1)
    bin_centres = 0.5 * (met_bins[:-1] + met_bins[1:])

    ylabel = r'$12 + \log_{10}\, (Z_{\mathrm{ICS}} / 0.02)$'
    xlabel = r'$12 + \log_{10}\, (Z_{\mathrm{BCG}} / 0.02)$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(log_Zbcg_gc, log_Zics_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(met_bins) - 1):
        in_bin = (log_Zbcg_gc >= met_bins[i]) & (log_Zbcg_gc < met_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Zics_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    lo_line = min(x_lo, y_lo)
    hi_line = max(x_hi, y_hi)
    ax_gc.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(log_Zbcg_iso, log_Zics_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(met_bins) - 1):
        in_bin = (log_Zbcg_iso >= met_bins[i]) & (log_Zbcg_iso < met_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Zics_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_log_Zbcg, all_log_Zics, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.plot([lo_line, hi_line], [lo_line, hi_line], color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity12_vs_BCG_Metallicity12_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_vs_concentration_contour(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log M_ICS vs halo Concentration.
    Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS vs Concentration 3-panel contour: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['Concentration'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make contour plots.')
        return

    conc_gc = d['Concentration'][gc_mask]
    log_Mics_gc = np.log10(d['ICS'][gc_mask])
    conc_iso = d['Concentration'][iso_mask]
    log_Mics_iso = np.log10(d['ICS'][iso_mask])

    all_conc = np.concatenate([conc_gc, conc_iso])
    all_log_Mics = np.concatenate([log_Mics_gc, log_Mics_iso])

    x_lo, x_hi = max(all_conc.min() - 0.5, 0), all_conc.max() + 0.5
    y_lo, y_hi = all_log_Mics.min() - 0.25, all_log_Mics.max() + 0.5

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    # Bin width ~ 1 for concentration
    conc_bin_w = max(0.5, (x_hi - x_lo) / 20)
    conc_bins = np.arange(x_lo, x_hi + conc_bin_w, conc_bin_w)
    bin_centres = 0.5 * (conc_bins[:-1] + conc_bins[1:])

    xlabel = r'Concentration'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(conc_gc, log_Mics_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(conc_bins) - 1):
        in_bin = (conc_gc >= conc_bins[i]) & (conc_gc < conc_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(log_Mics_gc[in_bin]))
            bin_x_gc.append(bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(conc_iso, log_Mics_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(conc_bins) - 1):
        in_bin = (conc_iso >= conc_bins[i]) & (conc_iso < conc_bins[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(log_Mics_iso[in_bin]))
            bin_x_iso.append(bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_conc, all_log_Mics, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_vs_Concentration_contour_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_vs_concentration_mvir_binned(file_list, sim_params, snap_num, output_dir):
    """
    2-panel (groups/clusters, isolated) plot of median log M_ICS vs Concentration,
    binned in Mvir to isolate the concentration dependence at fixed halo mass.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS vs Concentration (Mvir-binned): snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['Concentration'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    if np.sum(gc_mask) < 10 and np.sum(iso_mask) < 10:
        print('  Too few galaxies to make plot.')
        return

    # Mvir bins (log10 Mvir)
    mvir_edges = np.array([11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0])

    # Concentration bins
    all_conc = d['Concentration'][base_mask]
    x_lo, x_hi = max(all_conc.min() - 0.5, 0), all_conc.max() + 0.5
    conc_edges = np.linspace(x_lo, x_hi, 16)
    conc_centres = 0.5 * (conc_edges[:-1] + conc_edges[1:])

    fig, (ax_gc, ax_iso) = plt.subplots(1, 2, sharey=True, figsize=(16, 6.25))

    cmap_bins = plt.cm.viridis(np.linspace(0.05, 0.9, len(mvir_edges) - 1))

    def _binned_stats(pop_mask, ls):
        logM = np.log10(d['Mvir'][pop_mask])
        conc = d['Concentration'][pop_mask]
        logIcs = np.log10(d['ICS'][pop_mask])
        out = []
        for k in range(len(mvir_edges) - 1):
            m_in = (logM >= mvir_edges[k]) & (logM < mvir_edges[k+1])
            if np.sum(m_in) < 20:
                out.append(None)
                continue
            meds, los, his, xs = [], [], [], []
            for i in range(len(conc_edges) - 1):
                c_in = m_in & (conc >= conc_edges[i]) & (conc < conc_edges[i+1])
                if np.sum(c_in) >= 5:
                    meds.append(np.median(logIcs[c_in]))
                    los.append(np.percentile(logIcs[c_in], 16))
                    his.append(np.percentile(logIcs[c_in], 84))
                    xs.append(conc_centres[i])
            out.append((np.array(xs), np.array(meds), np.array(los), np.array(his)))
        return out

    stats_gc = _binned_stats(gc_mask, '-')
    stats_iso = _binned_stats(iso_mask, '-.')

    # --- Print per-bin diagnostics ---
    def _print_trend(stats_list, label):
        print(f'  --- {label} ---')
        for k, stats in enumerate(stats_list):
            lo, hi = mvir_edges[k], mvir_edges[k+1]
            if stats is None:
                print(f'    log M_vir [{lo:.1f},{hi:.1f}): too few haloes (<20)')
                continue
            xs, meds, los, his = stats
            if xs.size < 2:
                print(f'    log M_vir [{lo:.1f},{hi:.1f}): only {xs.size} conc. bin(s) filled')
                continue
            # Weighted log-log fit: slope of median log(M_ICS) vs concentration
            slope_c, _ = np.polyfit(xs, meds, 1)
            med_range = meds.max() - meds.min()
            print(f'    log M_vir [{lo:.1f},{hi:.1f}): {xs.size} conc. bins, '
                  f'c=[{xs[0]:.1f},{xs[-1]:.1f}], '
                  f'median log M_ICS=[{meds.min():.2f},{meds.max():.2f}] '
                  f'Δ={med_range:+.2f} dex, '
                  f'd<logM_ICS>/dc = {slope_c:+.3f} dex/unit')

    _print_trend(stats_gc, 'Groups/Clusters')
    _print_trend(stats_iso, 'Isolated')

    for k, (color, stats) in enumerate(zip(cmap_bins, stats_gc)):
        if stats is None:
            continue
        xs, meds, los, his = stats
        label = rf'$10^{{{mvir_edges[k]:.1f}}}$–$10^{{{mvir_edges[k+1]:.1f}}}\,\mathrm{{M}}_\odot$'
        ax_gc.fill_between(xs, los, his, color=color, alpha=0.2)
        ax_gc.plot(xs, meds, color=color, ls='-', lw=2.25, label=label)

    for k, (color, stats) in enumerate(zip(cmap_bins, stats_iso)):
        if stats is None:
            continue
        xs, meds, los, his = stats
        label = rf'$10^{{{mvir_edges[k]:.1f}}}$–$10^{{{mvir_edges[k+1]:.1f}}}\,\mathrm{{M}}_\odot$'
        ax_iso.fill_between(xs, los, his, color=color, alpha=0.2)
        ax_iso.plot(xs, meds, color=color, ls='-.', lw=2.25, label=label)

    ax_gc.set_xlabel(r'Concentration')
    ax_gc.set_ylabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$')
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_title(r'Groups/Clusters ($\geq 1$ sat.)', fontsize=12)
    ax_gc.legend(loc='lower right', fontsize=8, title=r'$M_{\mathrm{vir}}$')

    ax_iso.set_xlabel(r'Concentration')
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_title('Isolated (no sat.)', fontsize=12)
    ax_iso.legend(loc='lower right', fontsize=8, title=r'$M_{\mathrm{vir}}$')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_vs_Concentration_MvirBinned_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_mvir_residual_vs_concentration(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour plot of log(M_ICS) residual (from linear log-log fit against Mvir)
    vs Concentration. Panels: Groups/Clusters, Isolated, Combined.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS(Mvir) residual vs Concentration: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['Concentration'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few galaxies to fit.')
        return

    # Fit log(M_ICS) = slope * log(M_vir) + intercept on the combined sample
    logM = np.log10(d['Mvir'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    slope, intercept = np.polyfit(logM, logIcs, 1)
    print(f'  Fit: log(M_ICS) = {slope:.3f}*log(M_vir) + {intercept:.3f}')

    # Compute residuals for each mask (GC, ISO, combined)
    def _residual(mask):
        logM_ = np.log10(d['Mvir'][mask])
        return np.log10(d['ICS'][mask]) - (slope * logM_ + intercept)

    res_gc = _residual(gc_mask)
    res_iso = _residual(iso_mask)
    res_all = _residual(base_mask)

    conc_gc = d['Concentration'][gc_mask]
    conc_iso = d['Concentration'][iso_mask]
    conc_all = d['Concentration'][base_mask]

    print(f'  {np.sum(gc_mask)} group/cluster centrals, {np.sum(iso_mask)} isolated centrals')

    # --- Residual/concentration diagnostics ---
    def _res_stats(label, conc_arr, res_arr):
        if res_arr.size < 10:
            print(f'  [{label}] too few points ({res_arr.size})')
            return
        sl, _ = np.polyfit(conc_arr, res_arr, 1)
        rho = np.corrcoef(conc_arr, res_arr)[0, 1]
        print(f'  [{label}] N={res_arr.size}, conc=[{conc_arr.min():.2f},{conc_arr.max():.2f}] '
              f'(med {np.median(conc_arr):.2f}), '
              f'residual med={np.median(res_arr):+.3f}, scatter(σ)={np.std(res_arr):.3f} dex, '
              f'Pearson r={rho:+.3f}, d(res)/dc={sl:+.3f} dex/unit')
        # Quartile trend
        q = np.quantile(conc_arr, [0.25, 0.5, 0.75])
        lo_m = conc_arr < q[0]
        hi_m = conc_arr > q[2]
        if np.any(lo_m) and np.any(hi_m):
            print(f'    median residual: low-c quartile = {np.median(res_arr[lo_m]):+.3f}, '
                  f'high-c quartile = {np.median(res_arr[hi_m]):+.3f}, '
                  f'Δ = {np.median(res_arr[hi_m]) - np.median(res_arr[lo_m]):+.3f} dex')

    _res_stats('Groups/Clusters', conc_gc, res_gc)
    _res_stats('Isolated', conc_iso, res_iso)
    _res_stats('Combined', conc_all, res_all)

    x_lo, x_hi = max(conc_all.min() - 0.5, 0), conc_all.max() + 0.5
    y_lo = min(res_gc.min() if res_gc.size else 0, res_iso.min() if res_iso.size else 0) - 0.1
    y_hi = max(res_gc.max() if res_gc.size else 0, res_iso.max() if res_iso.size else 0) + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    conc_bin_w = max(0.5, (x_hi - x_lo) / 20)
    conc_bins = np.arange(x_lo, x_hi + conc_bin_w, conc_bin_w)
    bin_centres = 0.5 * (conc_bins[:-1] + conc_bins[1:])

    xlabel = r'Concentration'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ -\ \log_{10}\, M_{\mathrm{ICS}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'

    def _median_curve(conc_arr, res_arr):
        xs, ys = [], []
        for i in range(len(conc_bins) - 1):
            m = (conc_arr >= conc_bins[i]) & (conc_arr < conc_bins[i+1])
            if np.sum(m) >= 3:
                xs.append(bin_centres[i])
                ys.append(np.median(res_arr[m]))
        return xs, ys

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(conc_gc, res_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    bx_gc, by_gc = _median_curve(conc_gc, res_gc)
    if bx_gc:
        ax_gc.plot(bx_gc, by_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(conc_iso, res_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    bx_iso, by_iso = _median_curve(conc_iso, res_iso)
    if bx_iso:
        ax_iso.plot(bx_iso, by_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)
    ax_iso.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(conc_all, res_all, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bx_gc:
        ax_comb.plot(bx_gc, by_gc, color='black', ls='-', lw=2.25,
                     label='Groups/Clusters Median')
    if bx_iso:
        ax_comb.plot(bx_iso, by_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_MvirResidual_vs_Concentration_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_residual_vs_nsat(file_list, sim_params, snap_num, output_dir):
    """
    Contour of log(M_ICS) residual (from linear log-log fit vs Mvir) against
    number of satellites (current satellite count, a proxy for merger history).
    Single panel (isolated have n_sat=0 by construction; GC distribution spans
    the full x-range).
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS(Mvir) residual vs N_satellites: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar))
    if np.sum(base_mask) < 50:
        print('  Too few galaxies to fit.')
        return

    logM = np.log10(d['Mvir'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    slope, intercept = np.polyfit(logM, logIcs, 1)
    print(f'  Fit: log(M_ICS) = {slope:.3f}*log(M_vir) + {intercept:.3f}')

    res_all = logIcs - (slope * logM + intercept)
    nsat_all = d['n_satellites'][base_mask]
    print(f'  N={res_all.size} centrals, n_sat=[{nsat_all.min()},{nsat_all.max()}] (med {int(np.median(nsat_all))})')

    # Correlations (raw and log1p)
    sl_lin, _ = np.polyfit(nsat_all, res_all, 1)
    rho_lin = np.corrcoef(nsat_all, res_all)[0, 1]
    log_nsat = np.log10(nsat_all + 1)
    sl_log, _ = np.polyfit(log_nsat, res_all, 1)
    rho_log = np.corrcoef(log_nsat, res_all)[0, 1]
    print(f'  res vs n_sat:        Pearson r={rho_lin:+.3f}, slope={sl_lin:+.4f} dex/sat')
    print(f'  res vs log10(n_s+1): Pearson r={rho_log:+.3f}, slope={sl_log:+.3f} dex/dex')

    # Quartile comparison
    q = np.quantile(nsat_all, [0.25, 0.5, 0.75])
    lo_m = nsat_all <= q[0]
    hi_m = nsat_all >= q[2]
    if np.any(lo_m) and np.any(hi_m):
        print(f'    median residual: low-n_s quartile (n_s<={int(q[0])}) = {np.median(res_all[lo_m]):+.3f}, '
              f'high-n_s quartile (n_s>={int(q[2])}) = {np.median(res_all[hi_m]):+.3f}, '
              f'delta = {np.median(res_all[hi_m]) - np.median(res_all[lo_m]):+.3f} dex')

    # Plot residual vs log10(n_sat+1) contour
    x_lo, x_hi = 0, log_nsat.max() + 0.1
    y_lo = res_all.min() - 0.1
    y_hi = res_all.max() + 0.1

    x_bins = np.linspace(x_lo, x_hi, 60)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, ax = plt.subplots(figsize=(9, 6.25))

    H, xedges, yedges = np.histogram2d(log_nsat, res_all, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    H = H.T
    Hm = np.where(H > 0, H, np.nan)
    cf = ax.contourf(X, Y, Hm, levels=_int_levels(Hm), cmap='RdPu_r')

    # Binned median curve
    bw = max(0.1, (x_hi - x_lo) / 15)
    bins = np.arange(x_lo, x_hi + bw, bw)
    bcen = 0.5 * (bins[:-1] + bins[1:])
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (log_nsat >= bins[i]) & (log_nsat < bins[i+1])
        if np.sum(m) >= 5:
            xs.append(bcen[i])
            ys.append(np.median(res_all[m]))
    if xs:
        ax.plot(xs, ys, color='black', ls='-', lw=2.25, label='Median')

    ax.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log_{10}(N_{\mathrm{sat}} + 1)$')
    ax.set_ylabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ -\ \log_{10}\, M_{\mathrm{ICS}}^{\mathrm{fit}}(M_{\mathrm{vir}})$')
    fig.colorbar(cf, ax=ax).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_MvirResidual_vs_Nsat_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_residual_vs_bcg_residual(file_list, sim_params, snap_num, output_dir):
    """
    3-panel plot of (log M_ICS residual) vs (log M_BCG residual), both residuals
    computed from independent linear log-log fits against M_vir. If residuals
    correlate strongly, both ICS and BCG assembly are driven by the same
    halo-history variable.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS residual vs BCG residual (both off M_vir): snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few galaxies to fit.')
        return

    logM = np.log10(d['Mvir'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    logBcg = np.log10(d['StellarMass'][base_mask])

    sl_i, ic_i = np.polyfit(logM, logIcs, 1)
    sl_b, ic_b = np.polyfit(logM, logBcg, 1)
    print(f'  Fit ICS: log(M_ICS) = {sl_i:.3f}*log(M_vir) + {ic_i:.3f}')
    print(f'  Fit BCG: log(M_BCG) = {sl_b:.3f}*log(M_vir) + {ic_b:.3f}')

    def _residuals(mask):
        lm = np.log10(d['Mvir'][mask])
        ri = np.log10(d['ICS'][mask]) - (sl_i * lm + ic_i)
        rb = np.log10(d['StellarMass'][mask]) - (sl_b * lm + ic_b)
        return ri, rb

    ri_gc, rb_gc = _residuals(gc_mask)
    ri_iso, rb_iso = _residuals(iso_mask)
    ri_all, rb_all = _residuals(base_mask)

    def _rr_stats(label, rb, ri):
        if ri.size < 10:
            return
        sl, _ = np.polyfit(rb, ri, 1)
        rho = np.corrcoef(rb, ri)[0, 1]
        print(f'  [{label}] N={ri.size}: Pearson r={rho:+.3f}, d(ICS_res)/d(BCG_res)={sl:+.3f} dex/dex, '
              f'sigma(ICS_res)={np.std(ri):.3f}, sigma(BCG_res)={np.std(rb):.3f}')

    _rr_stats('Groups/Clusters', rb_gc, ri_gc)
    _rr_stats('Isolated', rb_iso, ri_iso)
    _rr_stats('Combined', rb_all, ri_all)

    x_lo = min(rb_gc.min() if rb_gc.size else 0, rb_iso.min() if rb_iso.size else 0) - 0.1
    x_hi = max(rb_gc.max() if rb_gc.size else 0, rb_iso.max() if rb_iso.size else 0) + 0.1
    y_lo = min(ri_gc.min() if ri_gc.size else 0, ri_iso.min() if ri_iso.size else 0) - 0.1
    y_hi = max(ri_gc.max() if ri_gc.size else 0, ri_iso.max() if ri_iso.size else 0) + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$\log_{10}\, M_{\mathrm{BCG}}\ -\ \log_{10}\, M_{\mathrm{BCG}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ -\ \log_{10}\, M_{\mathrm{ICS}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'

    def _draw(ax, xarr, yarr, ls, lbl):
        H, xedges, yedges = np.histogram2d(xarr, yarr, bins=[x_bins, y_bins])
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc)
        H = H.T
        Hm = np.where(H > 0, H, np.nan)
        cf = ax.contourf(X, Y, Hm, levels=_int_levels(Hm), cmap='RdPu_r')
        # Running median in BCG-residual bins
        bw = max(0.05, (x_hi - x_lo) / 20)
        mb = np.arange(x_lo, x_hi + bw, bw)
        mbc = 0.5 * (mb[:-1] + mb[1:])
        xs, ys = [], []
        for i in range(len(mb) - 1):
            m = (xarr >= mb[i]) & (xarr < mb[i+1])
            if np.sum(m) >= 5:
                xs.append(mbc[i])
                ys.append(np.median(yarr[m]))
        if xs:
            ax.plot(xs, ys, color='black', ls=ls, lw=2.25, label=lbl)
        ax.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
        ax.axvline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel)
        ax.legend(loc='lower right', fontsize=10)
        fig.colorbar(cf, ax=ax).set_label(r'Number of haloes')
        return xs, ys

    xg, yg = _draw(ax_gc, rb_gc, ri_gc, '-', r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.set_ylabel(ylabel)
    xi, yi = _draw(ax_iso, rb_iso, ri_iso, '-.', 'Isolated (no sat.)')
    _draw(ax_comb, rb_all, ri_all, '-', 'Combined')
    if xg:
        ax_comb.plot(xg, yg, color='black', ls='-', lw=2.25)
    if xi:
        ax_comb.plot(xi, yi, color='black', ls='-.', lw=2.25)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Residual_vs_BCG_Residual_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_to_bcg_ratio_vs_mvir(file_list, sim_params, snap_num, output_dir):
    """
    3-panel contour of log10(M_ICS / M_BCG) vs log10(M_vir). A flat relation
    across M_vir would indicate proportional growth (a single-timescale process);
    a tilt indicates mass-dependent merger/stripping efficiency.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nICS/BCG ratio vs M_vir: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few galaxies.')
        return

    logM_all = np.log10(d['Mvir'][base_mask])
    ratio_all = np.log10(d['ICS'][base_mask] / d['StellarMass'][base_mask])
    logM_gc = np.log10(d['Mvir'][gc_mask])
    ratio_gc = np.log10(d['ICS'][gc_mask] / d['StellarMass'][gc_mask])
    logM_iso = np.log10(d['Mvir'][iso_mask])
    ratio_iso = np.log10(d['ICS'][iso_mask] / d['StellarMass'][iso_mask])

    def _ratio_stats(label, logm, r):
        if r.size < 10:
            return
        sl, ic = np.polyfit(logm, r, 1)
        rho = np.corrcoef(logm, r)[0, 1]
        print(f'  [{label}] N={r.size}: med log(ICS/BCG)={np.median(r):+.3f}, sigma={np.std(r):.3f} dex, '
              f'slope d/dlogMvir={sl:+.3f}, intercept={ic:+.3f}, Pearson r={rho:+.3f}')
        for ml in [11.5, 12.0, 12.5, 13.0, 13.5, 14.0]:
            mm = (logm >= ml - 0.25) & (logm < ml + 0.25)
            if np.sum(mm) >= 5:
                print(f'    logMvir~{ml:.1f}: N={int(np.sum(mm))}, median log(ICS/BCG)={np.median(r[mm]):+.3f}')

    _ratio_stats('Combined', logM_all, ratio_all)
    _ratio_stats('Groups/Clusters', logM_gc, ratio_gc)
    _ratio_stats('Isolated', logM_iso, ratio_iso)

    x_lo = np.floor(logM_all.min() * 2) / 2
    x_hi = np.ceil(logM_all.max() * 2) / 2
    y_lo = ratio_all.min() - 0.1
    y_hi = ratio_all.max() + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$'
    ylabel = r'$\log_{10}\, (M_{\mathrm{ICS}} / M_{\mathrm{BCG}})$'

    bw = max(0.1, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)
    mbc = 0.5 * (mb[:-1] + mb[1:])

    def _median_curve(logm, r):
        xs, ys = [], []
        for i in range(len(mb) - 1):
            m = (logm >= mb[i]) & (logm < mb[i+1])
            if np.sum(m) >= 5:
                xs.append(mbc[i])
                ys.append(np.median(r[m]))
        return xs, ys

    # GC panel
    H_gc, xedges, yedges = np.histogram2d(logM_gc, ratio_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    Hm = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, Hm, levels=_int_levels(Hm), cmap='RdPu_r')
    xg_, yg_ = _median_curve(logM_gc, ratio_gc)
    if xg_:
        ax_gc.plot(xg_, yg_, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # Iso panel
    H_iso, _, _ = np.histogram2d(logM_iso, ratio_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    Hm = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, Hm, levels=_int_levels(Hm), cmap='RdPu_r')
    xi_, yi_ = _median_curve(logM_iso, ratio_iso)
    if xi_:
        ax_iso.plot(xi_, yi_, color='black', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)
    ax_iso.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # Combined panel
    H_comb, _, _ = np.histogram2d(logM_all, ratio_all, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    Hm = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, Hm, levels=_int_levels(Hm), cmap='RdPu_r')
    if xg_:
        ax_comb.plot(xg_, yg_, color='black', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_:
        ax_comb.plot(xi_, yi_, color='black', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_to_BCG_Ratio_vs_Mvir_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def _find_snapshots_for_redshifts(sim_params, z_min=0, z_max=3):
    """Find all available output snapshots with redshifts in [z_min, z_max]."""
    redshifts = sim_params['redshifts']
    output_snaps = sim_params['output_snapshots']
    snaps = []
    for s in output_snaps:
        s = int(s)
        z = redshifts[s]
        if z_min <= z <= z_max:
            snaps.append(s)
    return sorted(snaps, reverse=True)  # high-z first


def _snapshot_z_bins(sim_params, snap_list):
    """Build z-bin edges centred on each snapshot's redshift so contours have no gaps."""
    zvals = np.array([sim_params['redshifts'][s] for s in snap_list])
    zvals = np.sort(zvals)
    # Midpoints between consecutive snapshot redshifts
    mids = 0.5 * (zvals[:-1] + zvals[1:])
    # Edges: half-step before first, midpoints, half-step after last
    lo = max(zvals[0] - (mids[0] - zvals[0]), -0.05)
    hi = zvals[-1] + (zvals[-1] - mids[-1])
    edges = np.concatenate([[lo], mids, [hi]])
    return edges


def _load_multi_snapshot_ics(file_list, sim_params, snap_list):
    """Generator yielding ICS data for each snapshot in snap_list."""
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    for snap_num in snap_list:
        d = load_ics_data(file_list, sim_params, snap_num)
        if d is None:
            continue

        z = d['redshift']
        base_mask = (d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0) & (d['StellarMass'] >= min_stellar)
        gc_mask = base_mask & (d['n_satellites'] >= 1)
        iso_mask = base_mask & (d['n_satellites'] == 0)

        yield snap_num, z, d, gc_mask, iso_mask


def plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir):
    """
    3-panel contour plot of f_ICS = M_ICS / (f_b * Mvir) vs redshift (z=0-3).
    """
    baryon_frac = sim_params['BaryonFrac']
    snap_list = _find_snapshots_for_redshifts(sim_params, 0, 3)

    print(f'\nf_ICS(f_b*Mvir) vs redshift: {len(snap_list)} snapshots in z=0-3...')

    z_gc, fICS_gc = [], []
    z_iso, fICS_iso = [], []

    for snap_num, z, d, gc_mask, iso_mask in _load_multi_snapshot_ics(file_list, sim_params, snap_list):
        if np.sum(gc_mask) > 0:
            f = d['ICS'][gc_mask] / (baryon_frac * d['Mvir'][gc_mask])
            z_gc.append(np.full(np.sum(gc_mask), z))
            fICS_gc.append(np.log10(f))
        if np.sum(iso_mask) > 0:
            f = d['ICS'][iso_mask] / (baryon_frac * d['Mvir'][iso_mask])
            z_iso.append(np.full(np.sum(iso_mask), z))
            fICS_iso.append(np.log10(f))
        print(f'  z={z:.2f} (snap {snap_num}): {np.sum(gc_mask)} gc, {np.sum(iso_mask)} iso')

    if not z_gc and not z_iso:
        print('  No data found.')
        return

    z_gc = np.concatenate(z_gc) if z_gc else np.array([])
    fICS_gc = np.concatenate(fICS_gc) if fICS_gc else np.array([])
    z_iso = np.concatenate(z_iso) if z_iso else np.array([])
    fICS_iso = np.concatenate(fICS_iso) if fICS_iso else np.array([])

    all_z = np.concatenate([z_gc, z_iso])
    all_fICS = np.concatenate([fICS_gc, fICS_iso])

    x_lo, x_hi = -0.1, 3.1
    y_lo, y_hi = all_fICS.min() - 0.25, all_fICS.max() + 0.5
    # Use snapshot-based z bins so every snapshot fills a contiguous band
    x_bins = _snapshot_z_bins(sim_params, snap_list)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True,
                                                  figsize=(24, 6.25))

    # Redshift bins for medians
    z_bin_edges = np.linspace(0, 3, 25)
    z_bin_centres = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])

    xlabel = r'$z$'
    ylabel = r'$\log_{10}\, f_{\mathrm{ICS}}\ =\ m_{\mathrm{ICS}} / (f_b\ M_{\mathrm{vir}})$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(z_gc, fICS_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(z_bin_edges) - 1):
        in_bin = (z_gc >= z_bin_edges[i]) & (z_gc < z_bin_edges[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(fICS_gc[in_bin]))
            bin_x_gc.append(z_bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(z_iso, fICS_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(z_bin_edges) - 1):
        in_bin = (z_iso >= z_bin_edges[i]) & (z_iso < z_bin_edges[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(fICS_iso[in_bin]))
            bin_x_iso.append(z_bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_z, all_fICS, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label=r'Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_fraction_vs_redshift{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_stellar_fraction_vs_redshift(file_list, sim_params, output_dir):
    """
    3-panel contour plot of f_ICS = M_ICS / (M_*,halo + M_ICS) vs redshift (z=0-3).
    """
    snap_list = _find_snapshots_for_redshifts(sim_params, 0, 3)

    print(f'\nf_ICS(stellar) vs redshift: {len(snap_list)} snapshots in z=0-3...')

    z_gc, fICS_gc = [], []
    z_iso, fICS_iso = [], []

    for snap_num, z, d, gc_mask, iso_mask in _load_multi_snapshot_ics(file_list, sim_params, snap_list):
        if np.sum(gc_mask) > 0:
            f = d['ICS'][gc_mask] / (d['HaloStellarMass'][gc_mask] + d['ICS'][gc_mask])
            z_gc.append(np.full(np.sum(gc_mask), z))
            fICS_gc.append(np.log10(f))
        if np.sum(iso_mask) > 0:
            f = d['ICS'][iso_mask] / (d['HaloStellarMass'][iso_mask] + d['ICS'][iso_mask])
            z_iso.append(np.full(np.sum(iso_mask), z))
            fICS_iso.append(np.log10(f))
        print(f'  z={z:.2f} (snap {snap_num}): {np.sum(gc_mask)} gc, {np.sum(iso_mask)} iso')

    if not z_gc and not z_iso:
        print('  No data found.')
        return

    z_gc = np.concatenate(z_gc) if z_gc else np.array([])
    fICS_gc = np.concatenate(fICS_gc) if fICS_gc else np.array([])
    z_iso = np.concatenate(z_iso) if z_iso else np.array([])
    fICS_iso = np.concatenate(fICS_iso) if fICS_iso else np.array([])

    all_z = np.concatenate([z_gc, z_iso])
    all_fICS = np.concatenate([fICS_gc, fICS_iso])

    x_lo, x_hi = -0.1, 3.1
    y_lo, y_hi = all_fICS.min() - 0.25, all_fICS.max() + 0.5
    x_bins = _snapshot_z_bins(sim_params, snap_list)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True,
                                                  figsize=(24, 6.25))

    z_bin_edges = np.linspace(0, 3, 25)
    z_bin_centres = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])

    xlabel = r'$z$'
    ylabel = r'$\log_{10}\, f_{\mathrm{ICS}}\ =\ M_{\mathrm{ICS}} / (M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$'

    # --- Left panel: Groups & Clusters ---
    H_gc, xedges, yedges = np.histogram2d(z_gc, fICS_gc, bins=[x_bins, y_bins])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    H_gc = H_gc.T
    _Hm_gc = np.where(H_gc > 0, H_gc, np.nan)
    cf_gc = ax_gc.contourf(Xg, Yg, _Hm_gc, levels=_int_levels(_Hm_gc), cmap='RdPu_r')

    medians_gc, bin_x_gc = [], []
    for i in range(len(z_bin_edges) - 1):
        in_bin = (z_gc >= z_bin_edges[i]) & (z_gc < z_bin_edges[i+1])
        if np.sum(in_bin) >= 3:
            medians_gc.append(np.median(fICS_gc[in_bin]))
            bin_x_gc.append(z_bin_centres[i])
    if bin_x_gc:
        ax_gc.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')

    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    # --- Middle panel: Isolated ---
    H_iso, _, _ = np.histogram2d(z_iso, fICS_iso, bins=[x_bins, y_bins])
    H_iso = H_iso.T
    _Hm_iso = np.where(H_iso > 0, H_iso, np.nan)
    cf_iso = ax_iso.contourf(Xg, Yg, _Hm_iso, levels=_int_levels(_Hm_iso), cmap='RdPu_r')

    medians_iso, bin_x_iso = [], []
    for i in range(len(z_bin_edges) - 1):
        in_bin = (z_iso >= z_bin_edges[i]) & (z_iso < z_bin_edges[i+1])
        if np.sum(in_bin) >= 3:
            medians_iso.append(np.median(fICS_iso[in_bin]))
            bin_x_iso.append(z_bin_centres[i])
    if bin_x_iso:
        ax_iso.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                    label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)

    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    # --- Right panel: Combined ---
    H_comb, _, _ = np.histogram2d(all_z, all_fICS, bins=[x_bins, y_bins])
    H_comb = H_comb.T
    _Hm_comb = np.where(H_comb > 0, H_comb, np.nan)
    cf_comb = ax_comb.contourf(Xg, Yg, _Hm_comb, levels=_int_levels(_Hm_comb), cmap='RdPu_r')

    if bin_x_gc:
        ax_comb.plot(bin_x_gc, medians_gc, color='black', ls='-', lw=2.25,
                     label=r'Groups/Clusters Median')
    if bin_x_iso:
        ax_comb.plot(bin_x_iso, medians_iso, color='black', ls='-.', lw=2.25,
                     label='Isolated Median')
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_stellar_fraction_vs_redshift{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_combined_fractions_vs_mvir(file_list, sim_params, snap_num, output_dir):
    """
    Single-panel median+1sigma plot of BOTH f_ICS definitions vs log Mvir,
    for groups/clusters and isolated populations.
    """
    baryon_frac = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * baryon_frac
    redshift = sim_params['redshifts'][snap_num]
    print(f'\nCombined-definition fractions vs Mvir: snapshot {snap_num} (z = {redshift:.2f})...')

    d = load_ics_data(file_list, sim_params, snap_num)
    if d is None:
        print('  No data found.')
        return

    base_mask = (d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0) & (d['StellarMass'] >= min_stellar)
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    mass_bins = np.arange(11.0, 16.1, 0.2)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    def _binned_stats(logM, logf):
        xs, meds, los, his = [], [], [], []
        for i in range(len(mass_bins) - 1):
            in_bin = (logM >= mass_bins[i]) & (logM < mass_bins[i+1])
            if np.sum(in_bin) >= 5:
                xs.append(bin_centres[i])
                meds.append(np.median(logf[in_bin]))
                los.append(np.percentile(logf[in_bin], 16))
                his.append(np.percentile(logf[in_bin], 84))
        return np.array(xs), np.array(meds), np.array(los), np.array(his)

    # Definition 1: f_ICS = M_ICS / (f_b * Mvir)
    logM_gc = np.log10(d['Mvir'][gc_mask])
    logM_iso = np.log10(d['Mvir'][iso_mask])
    f1_gc = np.log10(d['ICS'][gc_mask] / (baryon_frac * d['Mvir'][gc_mask]))
    f1_iso = np.log10(d['ICS'][iso_mask] / (baryon_frac * d['Mvir'][iso_mask]))

    # Definition 2: f_ICS = M_ICS / (M_*,halo + M_ICS)
    f2_gc = np.log10(d['ICS'][gc_mask] / (d['HaloStellarMass'][gc_mask] + d['ICS'][gc_mask]))
    f2_iso = np.log10(d['ICS'][iso_mask] / (d['HaloStellarMass'][iso_mask] + d['ICS'][iso_mask]))

    x1_gc, m1_gc, lo1_gc, hi1_gc = _binned_stats(logM_gc, f1_gc)
    x1_iso, m1_iso, lo1_iso, hi1_iso = _binned_stats(logM_iso, f1_iso)
    x2_gc, m2_gc, lo2_gc, hi2_gc = _binned_stats(logM_gc, f2_gc)
    x2_iso, m2_iso, lo2_iso, hi2_iso = _binned_stats(logM_iso, f2_iso)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.25))

    c1 = 'tab:blue'    # f_b*Mvir definition
    c2 = 'tab:red'     # stellar definition
    alpha = 0.2

    if len(x1_gc):
        ax.fill_between(x1_gc, lo1_gc, hi1_gc, color=c1, alpha=alpha)
        ax.plot(x1_gc, m1_gc, color=c1, ls='-', lw=2.25,
                label=r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(f_b M_{\mathrm{vir}})$, GC')
    if len(x1_iso):
        ax.fill_between(x1_iso, lo1_iso, hi1_iso, color=c1, alpha=alpha)
        ax.plot(x1_iso, m1_iso, color=c1, ls='-.', lw=2.25,
                label=r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(f_b M_{\mathrm{vir}})$, Iso')
    if len(x2_gc):
        ax.fill_between(x2_gc, lo2_gc, hi2_gc, color=c2, alpha=alpha)
        ax.plot(x2_gc, m2_gc, color=c2, ls='-', lw=2.25,
                label=r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$, GC')
    if len(x2_iso):
        ax.fill_between(x2_iso, lo2_iso, hi2_iso, color=c2, alpha=alpha)
        ax.plot(x2_iso, m2_iso, color=c2, ls='-.', lw=2.25,
                label=r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$, Iso')

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\, f_{\mathrm{ICS}}$')
    ax.set_xlim(11.0, 16.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_combined_fractions_vs_Mvir_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_combined_fractions_vs_redshift(file_list, sim_params, output_dir):
    """
    Single-panel median+1sigma plot of BOTH f_ICS definitions vs redshift (z=0-3),
    for groups/clusters and isolated populations.
    """
    baryon_frac = sim_params['BaryonFrac']
    snap_list = _find_snapshots_for_redshifts(sim_params, 0, 3)

    print(f'\nCombined-definition fractions vs redshift: {len(snap_list)} snapshots in z=0-3...')

    zs = []
    # (median, p16, p84) per snapshot for each series
    m1_gc, lo1_gc, hi1_gc = [], [], []
    m1_iso, lo1_iso, hi1_iso = [], [], []
    m2_gc, lo2_gc, hi2_gc = [], [], []
    m2_iso, lo2_iso, hi2_iso = [], [], []

    for snap_num, z, d, gc_mask, iso_mask in _load_multi_snapshot_ics(file_list, sim_params, snap_list):
        zs.append(z)

        def _stats(arr):
            if arr.size < 5:
                return np.nan, np.nan, np.nan
            return np.median(arr), np.percentile(arr, 16), np.percentile(arr, 84)

        f1_gc_arr = np.log10(d['ICS'][gc_mask] / (baryon_frac * d['Mvir'][gc_mask])) if np.sum(gc_mask) else np.array([])
        f1_iso_arr = np.log10(d['ICS'][iso_mask] / (baryon_frac * d['Mvir'][iso_mask])) if np.sum(iso_mask) else np.array([])
        f2_gc_arr = np.log10(d['ICS'][gc_mask] / (d['HaloStellarMass'][gc_mask] + d['ICS'][gc_mask])) if np.sum(gc_mask) else np.array([])
        f2_iso_arr = np.log10(d['ICS'][iso_mask] / (d['HaloStellarMass'][iso_mask] + d['ICS'][iso_mask])) if np.sum(iso_mask) else np.array([])

        for arr, lst_m, lst_lo, lst_hi in (
            (f1_gc_arr, m1_gc, lo1_gc, hi1_gc),
            (f1_iso_arr, m1_iso, lo1_iso, hi1_iso),
            (f2_gc_arr, m2_gc, lo2_gc, hi2_gc),
            (f2_iso_arr, m2_iso, lo2_iso, hi2_iso),
        ):
            med, p16, p84 = _stats(arr)
            lst_m.append(med)
            lst_lo.append(p16)
            lst_hi.append(p84)

    zs = np.array(zs)

    def _arr(x):
        return np.array(x, dtype=float)

    m1_gc, lo1_gc, hi1_gc = _arr(m1_gc), _arr(lo1_gc), _arr(hi1_gc)
    m1_iso, lo1_iso, hi1_iso = _arr(m1_iso), _arr(lo1_iso), _arr(hi1_iso)
    m2_gc, lo2_gc, hi2_gc = _arr(m2_gc), _arr(lo2_gc), _arr(hi2_gc)
    m2_iso, lo2_iso, hi2_iso = _arr(m2_iso), _arr(lo2_iso), _arr(hi2_iso)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.25))

    c1 = 'tab:blue'
    c2 = 'tab:red'
    alpha = 0.2

    def _plot_series(x, med, lo, hi, color, ls, label):
        good = ~np.isnan(med)
        if good.sum() == 0:
            return
        ax.fill_between(x[good], lo[good], hi[good], color=color, alpha=alpha)
        ax.plot(x[good], med[good], color=color, ls=ls, lw=2.25, label=label)

    _plot_series(zs, m1_gc, lo1_gc, hi1_gc, c1, '-',
                 r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(f_b M_{\mathrm{vir}})$, GC')
    _plot_series(zs, m1_iso, lo1_iso, hi1_iso, c1, '-.',
                 r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(f_b M_{\mathrm{vir}})$, Iso')
    _plot_series(zs, m2_gc, lo2_gc, hi2_gc, c2, '-',
                 r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$, GC')
    _plot_series(zs, m2_iso, lo2_iso, hi2_iso, c2, '-.',
                 r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}/(M_{*,\mathrm{halo}} + M_{\mathrm{ICS}})$, Iso')

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log_{10}\, f_{\mathrm{ICS}}$')
    ax.set_xlim(0, 3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_combined_fractions_vs_redshift{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Contour plot of ICS fraction vs halo mass from SAGE26 output')

    parser.add_argument('input_pattern', nargs='?',
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files '
                             '(default: ./output/millennium/model_*.hdf5)')

    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest available)')

    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots '
                             '(default: <input_dir>/plots/)')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f'Error: No files found matching: {args.input_pattern}')
        sys.exit(1)

    print(f'Found {len(file_list)} model files.')

    sim_params = read_simulation_params(file_list[0])

    snap_num = args.snapshot if args.snapshot is not None else sim_params['last_snapshot']

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Baryon fraction: {sim_params["BaryonFrac"]:.4f}')
    print(f'Particle mass:   {sim_params["PartMass"]:.4e} Msun')
    print(f'Stellar mass threshold (PartMass * BaryonFrac): {sim_params["PartMass"] * sim_params["BaryonFrac"]:.4e} Msun')
    print(f'Output directory: {output_dir}\n')

    plot_ics_fraction_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_stellar_fraction_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_vs_bcg_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity12_vs_halostellar_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity_vs_bcg_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity12_vs_bcg_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity_vs_bcg_metallicity_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_metallicity12_vs_bcg_metallicity12_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_vs_concentration_contour(file_list, sim_params, snap_num, output_dir)

    plot_ics_vs_concentration_mvir_binned(file_list, sim_params, snap_num, output_dir)

    plot_ics_mvir_residual_vs_concentration(file_list, sim_params, snap_num, output_dir)

    plot_ics_residual_vs_nsat(file_list, sim_params, snap_num, output_dir)

    plot_ics_residual_vs_bcg_residual(file_list, sim_params, snap_num, output_dir)

    plot_ics_to_bcg_ratio_vs_mvir(file_list, sim_params, snap_num, output_dir)

    plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir)

    plot_ics_stellar_fraction_vs_redshift(file_list, sim_params, output_dir)

    plot_combined_fractions_vs_mvir(file_list, sim_params, snap_num, output_dir)

    plot_combined_fractions_vs_redshift(file_list, sim_params, output_dir)


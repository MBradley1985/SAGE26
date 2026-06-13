#!/usr/bin/env python
"""
ICS fraction analysis: extracts 1000 random groups/clusters between z=0 and z=0.5
and plots two definitions of the ICS fraction as a function of halo mass.
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import warnings
from scipy.stats import spearmanr, t

warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.frameon'] = False


def read_simulation_params(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['PartMass'] = float(sim.attrs['particle_mass']) * 1.0e10 / float(sim.attrs['hubble_h'])

        runtime = f['Header/Runtime']
        params['BaryonFrac'] = float(runtime.attrs.get('BaryonFrac', 0.17))

        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        
        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(int(s.replace('Snap_', '')) for s in snap_groups)
    return params


def _concat_field(file_list, snap_name, field, hubble_h, is_mass=False):
    pieces = []
    for fp in file_list:
        with h5.File(fp, 'r') as f:
            if snap_name not in f or field not in f[snap_name]:
                continue
            data = np.array(f[snap_name][field])
            if data.size > 0:
                if is_mass:
                    data = data * 1.0e10 / hubble_h
                pieces.append(data)
    if not pieces:
        return np.array([])
    return np.concatenate(pieces, axis=0)


def _load_snap_fields(file_list, snap_num, hubble_h, mass_fields, other_fields):
    snap_name = f'Snap_{snap_num}'
    data = {}
    for field in mass_fields:
        data[field] = _concat_field(file_list, snap_name, field, hubble_h, is_mass=True)
    for field in other_fields:
        data[field] = _concat_field(file_list, snap_name, field, hubble_h, is_mass=False)
    return data


def _compute_satellite_sums(Type, StellarMass, GalaxyIndex, CentralGalaxyIndex):
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    sat_sm = StellarMass[sat_mask]
    ins = np.searchsorted(sorted_gids, sat_central_gids)
    ins = np.clip(ins, 0, len(sorted_gids) - 1)
    valid = sorted_gids[ins] == sat_central_gids
    c_idx = np.where(valid, sorted_idx[ins], -1)
    ok = c_idx >= 0
    n_sat = np.zeros(len(Type), dtype=int)
    np.add.at(n_sat, c_idx[ok], 1)
    sm_sat = np.zeros(len(Type), dtype=float)
    np.add.at(sm_sat, c_idx[ok], sat_sm[ok])
    return n_sat, sm_sat


def extract_and_sample_haloes(sim_params, file_list, h_, f_b, min_stellar, n_target=1000, seed=42, target_mu=13.5, target_sigma=0.5):
    """
    Scans snapshots between z=0 and z=0.5, loads galaxy data, computes ICS fractions, 
    and returns a Gaussian-weighted sample of haloes based on log10(Mvir).
    
    Parameters:
    - target_mu: The mean (peak) of the Gaussian distribution for log10(Mvir).
    - target_sigma: The standard deviation (spread) of the Gaussian distribution.
    """
    redshifts = sim_params['redshifts']
    valid_snaps = [s for s in sim_params['available_snapshots'] 
                   if s < len(redshifts) and 0.0 <= redshifts[s] <= 0.5]

    print(f"Scanning {len(valid_snaps)} snapshots between z=0 and z=0.5...")

    all_mvir = []
    all_f_baryon = []
    all_f_stellar = []
    all_z = []

    for snap in valid_snaps:
        z = redshifts[snap]
        d = _load_snap_fields(file_list, snap, h_,
                              mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
                              other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
        
        if d['Mvir'].size == 0:
            continue
            
        n_sat, sm_sat = _compute_satellite_sums(d['Type'], d['StellarMass'], 
                                                d['GalaxyIndex'], d['CentralGalaxyIndex'])
        
        # Mask criteria: Centrals, >=1 sat, Mvir >= 10^12.5, BCG mass >= threshold, ICS > 0
        mask = ((d['Type'] == 0) & 
                (n_sat >= 1) & 
                (d['Mvir'] >= 10**12.5) & 
                (d['StellarMass'] >= min_stellar) & 
                (d['IntraClusterStars'] > 0))
        
        sel = np.where(mask)[0]
        if len(sel) == 0:
            continue
            
        mvir_sel = d['Mvir'][sel]
        ics_sel = d['IntraClusterStars'][sel]
        bcg_sm_sel = d['StellarMass'][sel]
        sat_sm_sel = sm_sat[sel]
        
        # Calculate fractions
        f_baryon = ics_sel / (f_b * mvir_sel)
        total_stellar = bcg_sm_sel + sat_sm_sel + ics_sel
        f_stellar = ics_sel / total_stellar
        
        all_mvir.extend(mvir_sel)
        all_f_baryon.extend(f_baryon)
        all_f_stellar.extend(f_stellar)
        all_z.extend(np.full(len(sel), z))

    all_mvir = np.array(all_mvir)
    all_f_baryon = np.array(all_f_baryon)
    all_f_stellar = np.array(all_f_stellar)
    all_z = np.array(all_z)

    total_candidates = len(all_mvir)
    print(f"Found {total_candidates} haloes matching criteria.")

    if total_candidates == 0:
        print("No candidates found to plot.")
        return None, None, None, None, 0

    # =========================================================
    # GAUSSIAN WEIGHTED SAMPLING
    # =========================================================
    n_select = min(n_target, total_candidates)
    
    # 1. Get the log10 masses to base our Gaussian on
    log_mvir_all = np.log10(all_mvir)
    
    # 2. Calculate the Gaussian PDF weight for each halo
    # Formula: e^(-0.5 * ((x - mu) / sigma)^2)
    weights = np.exp(-0.5 * ((log_mvir_all - target_mu) / target_sigma)**2)
    
    # 3. Normalize the weights so they sum exactly to 1.0 (required by numpy)
    probabilities = weights / np.sum(weights)

    # 4. Sample using the probabilities
    rng = np.random.default_rng(seed)  
    try:
        idx = rng.choice(total_candidates, size=n_select, replace=False, p=probabilities)
    except ValueError as e:
        # Fallback if the Gaussian is too narrow and we run out of valid candidates
        print("Warning: Gaussian is too narrow to sample without replacement. Falling back to replace=True.")
        idx = rng.choice(total_candidates, size=n_select, replace=True, p=probabilities)

    mvir_plot = all_mvir[idx]
    fb_plot = all_f_baryon[idx]
    fs_plot = all_f_stellar[idx]
    z_plot = all_z[idx]

    return mvir_plot, fb_plot, fs_plot, z_plot, n_select

def partial_spearman(x, y, covar):
    """
    Calculates the partial Spearman correlation between x and y, controlling for covar.
    Returns the correlation coefficient and the 2-tailed p-value.
    """
    # 1. Calculate the pairwise standard Spearman correlations
    r_xy, _ = spearmanr(x, y)
    r_xc, _ = spearmanr(x, covar)
    r_yc, _ = spearmanr(y, covar)
    
    # 2. Apply the partial correlation formula
    num = r_xy - (r_xc * r_yc)
    den = np.sqrt((1 - r_xc**2) * (1 - r_yc**2))
    
    if den == 0:
        return 0.0, 1.0
        
    r_partial = num / den
    r_partial = np.clip(r_partial, -1.0, 1.0) # Handle float precision edges
    
    # 3. Calculate the p-value (degrees of freedom = N - 3 for 1 covariate)
    n = len(x)
    df = n - 3
    
    if abs(r_partial) == 1.0:
        return r_partial, 0.0
        
    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2))
    p_val = 2 * t.sf(np.abs(t_stat), df) 
    
    return r_partial, p_val


def plot_ics_fractions(mvir, fb, fs, z, n_select, output_dir):
    """
    Plots the baryon and stellar ICS fractions side-by-side as a function of halo mass.
    Each subplot includes its own statistics text box.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    x_data = np.log10(mvir)

    # ==========================================
    # SUBPLOT 1: BARYON FRACTION
    # ==========================================
    sc1 = ax1.scatter(x_data, fb, c=z, cmap='plasma', 
                      marker='o', s=25, alpha=0.7, vmin=0, vmax=0.5)
    
    ax1.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax1.set_ylabel(r'$f_{\mathrm{ICS, baryon}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$')
    ax1.set_xlim(12.5, 14)
    
    med_b = np.median(fb)
    std_b = np.std(fb)
    corr_b, p_val_b = spearmanr(x_data, fb)
    p_corr_b, pp_val_b = partial_spearman(x_data, fb, z)
    
    stats_text_b = (
        f"Median (f_ICS): {med_b:.3f}   |   Std: {std_b:.3f}\n"
        f"Spearman r: {corr_b:.3f} (p={p_val_b:.2e})\n"
        f"Partial Spearman r (control z): {p_corr_b:.3f} (p={pp_val_b:.2e})"
    )
    
    ax1.text(0.5, -0.22, stats_text_b, 
             transform=ax1.transAxes, fontsize=11, ha='center', va='top',      
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='grey', alpha=0.9))

    # ==========================================
    # SUBPLOT 2: STELLAR FRACTION
    # ==========================================
    sc2 = ax2.scatter(x_data, fs, c=z, cmap='plasma', 
                      marker='o', s=25, alpha=0.7, vmin=0, vmax=0.5)
    
    ax2.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax2.set_ylabel(r'$f_{\mathrm{ICS, stellar}} = M_{\mathrm{ICS}} / (M_{\mathrm{total\_stellar}})$')
    ax2.set_xlim(12.5, 14)
    
    med_s = np.median(fs)
    std_s = np.std(fs)
    corr_s, p_val_s = spearmanr(x_data, fs)
    p_corr_s, pp_val_s = partial_spearman(x_data, fs, z)
    
    stats_text_s = (
        f"Median (f_ICS): {med_s:.3f}   |   Std: {std_s:.3f}\n"
        f"Spearman r: {corr_s:.3f} (p={p_val_s:.2e})\n"
        f"Partial Spearman r (control z): {p_corr_s:.3f} (p={pp_val_s:.2e})"
    )
    
    ax2.text(0.5, -0.22, stats_text_s, 
             transform=ax2.transAxes, fontsize=11, ha='center', va='top',      
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='grey', alpha=0.9))

    # ==========================================
    # SHARED FORMATTING & COLORBAR FIX
    # ==========================================
    fig.suptitle(rf'Random Sample of {n_select} Groups/Clusters ($0 \leq z \leq 0.5$)', fontsize=15)
    
    # 1. TIGHT LAYOUT FIRST: Leave 8% of the figure empty on the right side (0.92 limit)
    plt.tight_layout(rect=[0, 0.12, 0.92, 0.95]) 
    
    # 2. ADD COLORBAR AXIS: [left, bottom, width, height] in relative figure coordinates
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) 
    
    # 3. DRAW COLORBAR in the safe space
    cbar = fig.colorbar(sc2, cax=cbar_ax)
    cbar.set_label('Redshift $z$')
    
    outfile = os.path.join(output_dir, 'ICS_Fractions_SideBySide.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to: {outfile}')

def plot_halo_distributions(mvir, z, n_select, output_dir):
    """
    Plots the distributions of redshift and halo mass for the selected sample side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ==========================================
    # SUBPLOT 1: REDSHIFT DISTRIBUTION
    # ==========================================
    # Using specific bins between 0 and 0.5 to keep it clean
    ax1.hist(z, bins=np.linspace(0, 0.5, 16), color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel('Number of Haloes')
    ax1.set_xlim(-0.02, 0.52)
    
    # ==========================================
    # SUBPLOT 2: HALO MASS DISTRIBUTION
    # ==========================================
    log_mvir = np.log10(mvir)
    ax2.hist(log_mvir, bins=20, color='firebrick', edgecolor='black', alpha=0.8)
    ax2.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax2.set_ylabel('Number of Haloes')
    
    # ==========================================
    # SHARED FORMATTING
    # ==========================================
    fig.suptitle(rf'Demographics of the {n_select} Randomly Sampled Groups/Clusters', fontsize=15)
    plt.tight_layout()

    outfile = os.path.join(output_dir, 'ICS_Sample_Distributions.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved distribution plot to: {outfile}')


def main():
    parser = argparse.ArgumentParser(description='Plot ICS fractions for 1000 random haloes (0 <= z <= 0.5)')
    parser.add_argument('input_pattern', nargs='?', default='./output/microuchuu/model_*.hdf5',
                        help='Path pattern to model HDF5 files')
    parser.add_argument('-o', '--output-dir', type=str, default='./plots',
                        help='Output directory for the plot')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f'Error: no files match {args.input_pattern}')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    sim_params = read_simulation_params(file_list[0])
    
    h_ = sim_params['Hubble_h']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * f_b

    ## Identify and sample haloes with a Gaussian distribution around 13.5 logMvir
    mvir_plot, fb_plot, fs_plot, z_plot, n_select = extract_and_sample_haloes(
        sim_params=sim_params,
        file_list=file_list,
        h_=h_,
        f_b=f_b,
        min_stellar=min_stellar,
        n_target=177,
        target_mu=13.5,     # Peak of the Gaussian (log10 M_vir)
        target_sigma=0.3    # Tweak this to make the bell curve wider or narrower
    )

    if n_select == 0:
        return

    # Call the plotting function
    plot_ics_fractions(mvir_plot, fb_plot, fs_plot, z_plot, n_select, args.output_dir)
    plot_halo_distributions(mvir_plot, z_plot, n_select, args.output_dir)


if __name__ == '__main__':
    main()
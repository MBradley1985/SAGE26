    #!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.stats import gaussian_kde, stats
from random import sample, seed
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
DirName = './output/millennium/'
ObsDataDir = './data/Gas/'
ObsDataDir2 = './data/MZR/'
FileName = 'model_0.hdf5'
Snapshot = 'Snap_63'

# Simulation details
Hubble_h = 0.73        # Hubble parameter
BoxSize = 62.5         # h-1 Mpc
VolumeFraction = 1.0   # Fraction of the full volume output by the model

# Plotting options
whichimf = 1        # 0=Slapeter; 1=Chabrier
dilute = 100000       # Number of galaxies to plot in scatter plots
sSFRcut = -11.0     # Divide quiescent from star forming galaxies

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34,6.25)
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


# ==================================================================

def read_hdf(filename = None, snap_num = None, param = None):

    if filename is None:
        filename = DirName + FileName
    property = h5.File(filename,'r')
    return np.array(property[snap_num][param])

def read_obs_data(filename):
    """Read observational data files"""
    filepath = os.path.join(ObsDataDir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: Observational data file {filename} not found")
        return None
    
    data = np.loadtxt(filepath)
    return data


# ==================================================================

if __name__ == '__main__':

    print('Running allresults (local)\n')

    seed(2222)
    volume = (BoxSize/Hubble_h)**3.0 * VolumeFraction

    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)

    # Read galaxy properties
    print('Reading galaxy properties from', DirName+FileName)

    CentralMvir = read_hdf(snap_num = Snapshot, param = 'CentralMvir') * 1.0e10 / Hubble_h
    Mvir = read_hdf(snap_num = Snapshot, param = 'Mvir') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
    MetalsStellarMass = read_hdf(snap_num = Snapshot, param = 'MetalsStellarMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(snap_num = Snapshot, param = 'BulgeMass') * 1.0e10 / Hubble_h
    BlackHoleMass = read_hdf(snap_num = Snapshot, param = 'BlackHoleMass') * 1.0e10 / Hubble_h
    ColdGas = read_hdf(snap_num = Snapshot, param = 'ColdGas') * 1.0e10 / Hubble_h
    MetalsColdGas = read_hdf(snap_num = Snapshot, param = 'MetalsColdGas') * 1.0e10 / Hubble_h
    MetalsEjectedMass = read_hdf(snap_num = Snapshot, param = 'MetalsEjectedMass') * 1.0e10 / Hubble_h
    HotGas = read_hdf(snap_num = Snapshot, param = 'HotGas') * 1.0e10 / Hubble_h
    MetalsHotGas = read_hdf(snap_num = Snapshot, param = 'MetalsHotGas') * 1.0e10 / Hubble_h
    EjectedMass = read_hdf(snap_num = Snapshot, param = 'EjectedMass') * 1.0e10 / Hubble_h
    CGMgas = read_hdf(snap_num = Snapshot, param = 'CGMgas') * 1.0e10 / Hubble_h
    MetalsCGMgas = read_hdf(snap_num = Snapshot, param = 'MetalsCGMgas') * 1.0e10 / Hubble_h

    IntraClusterStars = read_hdf(snap_num = Snapshot, param = 'IntraClusterStars') * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(snap_num = Snapshot, param = 'DiskRadius')
    BulgeRadius = read_hdf(snap_num = Snapshot, param = 'BulgeRadius')

    AnnuliiRadius = read_hdf(snap_num = Snapshot, param = 'DiscRadii')
    Disc_gas = read_hdf(snap_num = Snapshot, param = 'DiscGas') * 1.0e10 / Hubble_h
    Disc_stars = read_hdf(snap_num = Snapshot, param = 'DiscStars') * 1.0e10 / Hubble_h
    Disc_sfr = read_hdf(snap_num = Snapshot, param = 'DiscSFR')
    Disc_h2 = read_hdf(snap_num = Snapshot, param = 'DiscH2') * 1.0e10 / Hubble_h
    Disc_dust = read_hdf(snap_num = Snapshot, param = 'DiscDust') * 1.0e10 / Hubble_h
    Disc_GasMetals = read_hdf(snap_num = Snapshot, param = 'DiscGasMetals') * 1.0e10 / Hubble_h
    Disc_StarsMetals = read_hdf(snap_num = Snapshot, param = 'DiscStarsMetals') * 1.0e10 / Hubble_h

    H2gas = read_hdf(snap_num = Snapshot, param = 'H2gas') * 1.0e10 / Hubble_h
    H1gas = read_hdf(snap_num = Snapshot, param = 'H1gas') * 1.0e10 / Hubble_h
    Vvir = read_hdf(snap_num = Snapshot, param = 'Vvir')
    Vmax = read_hdf(snap_num = Snapshot, param = 'Vmax')
    Rvir = read_hdf(snap_num = Snapshot, param = 'Rvir')
    SfrDisk = read_hdf(snap_num = Snapshot, param = 'SfrDisk')
    SfrBulge = read_hdf(snap_num = Snapshot, param = 'SfrBulge')

    CentralGalaxyIndex = read_hdf(snap_num = Snapshot, param = 'CentralGalaxyIndex')
    Type = read_hdf(snap_num = Snapshot, param = 'Type')
    Posx = read_hdf(snap_num = Snapshot, param = 'Posx')
    Posy = read_hdf(snap_num = Snapshot, param = 'Posy')
    Posz = read_hdf(snap_num = Snapshot, param = 'Posz')

    OutflowRate = read_hdf(snap_num = Snapshot, param = 'OutflowRate')
    MassLoading = read_hdf(snap_num = Snapshot, param = 'MassLoading')
    Cooling = read_hdf(snap_num = Snapshot, param = 'Cooling')
    Tvir = 35.9 * (Vvir)**2  # in Kelvin


    w = np.where(StellarMass > 1.0e10)[0]
    print('Number of galaxies read:', len(StellarMass))
    print('Galaxies more massive than 10^10 h-1 Msun:', len(w), '\n')

    # ==================================================================
    # PLOTTING GRID
    # ==================================================================
    print('Generating plotting grid...')

    # Set up a 3x2 grid of plots
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"SAGE Output Overview: {Snapshot}", fontsize=20, y=0.92)

    # Calculate Total SFR
    TotalSFR = SfrDisk + SfrBulge

    # --- Filter data for global scatter plots (mass > 10^8) ---
    mask_global = (StellarMass > 1e8) & (Mvir > 0)
    sm_global = StellarMass[mask_global]
    mvir_global = Mvir[mask_global]
    sfr_global = TotalSFR[mask_global]
    cg_global = ColdGas[mask_global]

    # If dilute is set, randomly sample the arrays to speed up scatter plotting
    if len(sm_global) > dilute:
        idx_sample = sample(range(len(sm_global)), dilute)
    else:
        idx_sample = range(len(sm_global))

    # ------------------------------------------------------------------
    # Plot 1: Stellar Mass-Halo Mass Relation (SMHM)
    # ------------------------------------------------------------------
    ax = axs[0, 0]
    ax.scatter(np.log10(mvir_global[idx_sample]), np.log10(sm_global[idx_sample]), 
               s=2, alpha=0.3, color='steelblue')
    ax.set_title('Stellar Mass - Halo Mass Relation')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / h^{-1}M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_* / h^{-1}M_\odot)$')
    ax.grid(True, alpha=0.3, ls='--')

    # ------------------------------------------------------------------
    # Plot 2: Main Sequence of Star Formation
    # ------------------------------------------------------------------
    ax = axs[0, 1]
    # Filter out zero-SFR galaxies for the log plot
    mask_sf = sfr_global[idx_sample] > 0
    ax.scatter(np.log10(sm_global[idx_sample][mask_sf]), np.log10(sfr_global[idx_sample][mask_sf]), 
               s=2, alpha=0.3, color='forestgreen')
    ax.set_title('Star Formation Main Sequence')
    ax.set_xlabel(r'$\log_{10}(M_* / h^{-1}M_\odot)$')
    ax.set_ylabel(r'$\log_{10}({\rm SFR} / M_\odot {\rm yr}^{-1})$')
    ax.grid(True, alpha=0.3, ls='--')

    # ------------------------------------------------------------------
    # Plot 3: Cold Gas Fraction vs Stellar Mass
    # ------------------------------------------------------------------
    ax = axs[1, 0]
    gas_frac = cg_global[idx_sample] / sm_global[idx_sample]
    mask_gas = gas_frac > 0
    ax.scatter(np.log10(sm_global[idx_sample][mask_gas]), np.log10(gas_frac[mask_gas]), 
               s=2, alpha=0.3, color='purple')
    ax.set_title('Cold Gas Fraction')
    ax.set_xlabel(r'$\log_{10}(M_* / h^{-1}M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm cold} / M_*)$')
    ax.set_ylim(-4, 2)
    ax.grid(True, alpha=0.3, ls='--')

    # ------------------------------------------------------------------
    # Plot 4: Global Baryonic Reservoirs Histogram
    # ------------------------------------------------------------------
    ax = axs[1, 1]
    # Create a histogram of the total mass in different phases
    bins = np.linspace(8, 14, 40)
    ax.hist(np.log10(StellarMass[StellarMass > 0]), bins=bins, histtype='step', lw=2, color='orange', label='Stellar')
    ax.hist(np.log10(ColdGas[ColdGas > 0]), bins=bins, histtype='step', lw=2, color='blue', label='Cold Gas')
    ax.hist(np.log10(HotGas[HotGas > 0]), bins=bins, histtype='step', lw=2, color='red', label='Hot Gas')
    ax.hist(np.log10(EjectedMass[EjectedMass > 0]), bins=bins, histtype='step', lw=2, color='gray', label='Ejected')
    
    ax.set_title('Mass Functions of Baryonic Reservoirs')
    ax.set_xlabel(r'$\log_{10}(M / h^{-1}M_\odot)$')
    ax.set_ylabel(r'Number of Galaxies')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')

    # ------------------------------------------------------------------
    # SELECT A MASSIVE STAR-FORMING GALAXY FOR RADIAL PLOTS
    # ------------------------------------------------------------------
    valid_rad_gals = np.where((StellarMass > 5e10) & (TotalSFR > 2.0))[0]
    
    if len(valid_rad_gals) > 0:
        gal_idx = valid_rad_gals[0] 
        
        # --- FIXED RADIAL LOGIC ---
        N_BINS = 30
        # 1. Get edges and convert to kpc. Ensure we only take the first 31 edges.
        r_edges = AnnuliiRadius[gal_idx, :N_BINS+1] * 1000.0 / Hubble_h 
        r_in = r_edges[:-1]
        r_out = r_edges[1:]

        # 2. Calculate Areas
        area = np.pi * (r_out**2 - r_in**2)

        # 3. CREATE MASK
        # Ignore R < 0.1 kpc to remove the central numerical spike.
        # Ignore bins with zero mass or area to avoid log errors.
        mask = (r_in >= 0.1) & (area > 0) & (Disc_gas[gal_idx, :N_BINS] > 0)

        # Apply mask to everything
        r_plot = r_in[mask]
        area_plot = area[mask]
        
        # Surface Densities
        sig_stars = Disc_stars[gal_idx, :N_BINS][mask] / area_plot
        sig_gas = Disc_gas[gal_idx, :N_BINS][mask] / area_plot
        sig_h2 = Disc_h2[gal_idx, :N_BINS][mask] / area_plot
        sig_sfr = Disc_sfr[gal_idx, :N_BINS][mask] / area_plot
        sig_dust = Disc_dust[gal_idx, :N_BINS][mask] / area_plot
        sig_gas_metals = Disc_GasMetals[gal_idx, :N_BINS][mask] / area_plot
        sig_star_metals = Disc_StarsMetals[gal_idx, :N_BINS][mask] / area_plot

        # ------------------------------------------------------------------
        # Plot 5: Resolved Surface Density Profile
        # ------------------------------------------------------------------
        ax = axs[2, 0]
        ax.step(r_plot, sig_stars, where='post', lw=2, color='orange', label='Stars')
        ax.step(r_plot, sig_gas, where='post', lw=2, color='blue', label='Total Cold Gas')
        ax.step(r_plot, sig_h2, where='post', lw=2, ls='--', color='purple', label='H2 Gas')
        ax.step(r_plot, sig_sfr * 1e9, where='post', lw=2, color='green', label=r'SFR $\times 10^9$')

        ax.set_title(f'Radial Profile (Index: {gal_idx})')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel(r'Surface Density [$M_\odot$ kpc$^{-2}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.1, r_plot[-1] * 1.1) 
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # Plot 6: Resolved Chemistry (Metals & Dust)
        # ------------------------------------------------------------------
        ax = axs[2, 1]
        ax.step(r_plot, sig_dust, where='post', lw=2, color='brown', label='Dust')
        ax.step(r_plot, sig_gas_metals, where='post', lw=2, color='cyan', label='Gas Metals')
        ax.step(r_plot, sig_star_metals, where='post', lw=2, color='magenta', label='Stellar Metals')

        ax.set_title('Chemical Profile')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel(r'Surface Density [$M_\odot$ kpc$^{-2}$]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.1, r_plot[-1] * 1.1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    else:
        print("Warning: Could not find a massive star-forming galaxy for radial plots.")
        axs[2, 0].text(0.5, 0.5, 'No valid galaxies found\nfor radial plots', ha='center')
        axs[2, 1].text(0.5, 0.5, 'No valid galaxies found\nfor radial plots', ha='center')

    # Save and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_file = OutputDir + 'SAGE_Grid_Overview' + OutputFormat
    plt.savefig(out_file)
    print(f'Grid saved to {out_file}')
    plt.close()
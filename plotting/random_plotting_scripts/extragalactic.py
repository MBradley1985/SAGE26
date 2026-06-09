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
dilute = 7500       # Number of galaxies to plot in scatter plots
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

    HotGasDust = read_hdf(snap_num = Snapshot, param = 'HotDust') * 1.0e10 / Hubble_h
    CGMgasDust = read_hdf(snap_num = Snapshot, param = 'CGMDust') * 1.0e10 / Hubble_h
    EjectedMassDust = read_hdf(snap_num = Snapshot, param = 'EjectedDust') * 1.0e10 / Hubble_h


    w = np.where(StellarMass > 1.0e10)[0]
    print('Number of galaxies read:', len(StellarMass))
    print('Galaxies more massive than 10^10 h-1 Msun:', len(w), '\n')

# -------------------------------------------------------------------------

    print('Plotting extra-galactic mass/gas + dust vs halo mass')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    Mvir_cen = Mvir[w_cen]
    HotGas_cen = HotGas[w_cen]
    ICS_cen = IntraClusterStars[w_cen]
    CGMgas_cen = CGMgas[w_cen]
    Dust_cen = HotGasDust[w_cen] + CGMgasDust[w_cen] + EjectedMassDust[w_cen]

    # Define bins (logarithmic for both axes)
    x_bins = np.logspace(10, 15, 250)
    y_bins = np.logspace(-20, 14, 250)

    # List of (component, label, colormap)
    components = [
        (CGMgas_cen, 'CGM', 'Greens_r'),
        (ICS_cen, 'ICS', 'plasma'),
        (Dust_cen, 'Dust', 'Blues_r'),
        (HotGas_cen, 'IGM/ICM', 'PuRd_r')
    ]

    from scipy.ndimage import gaussian_filter


    # Plot filled contours and collect legend handles manually
    from matplotlib.patches import Patch
    legend_handles = []
    for comp, label, cmap in components:
        # 2D histogram for number count
        H, xedges, yedges = np.histogram2d(Mvir_cen, comp, bins=[x_bins, y_bins])
        # Convert to number density per dex^2
        # Bin widths in log10 space
        dx = np.diff(np.log10(xedges))
        dy = np.diff(np.log10(yedges))
        bin_area = dx[:, None] * dy[None, :]  # shape (nx, ny)
        # Avoid division by zero
        bin_area[bin_area == 0] = np.nan
        density = H / bin_area  # number per dex^2
        density = np.nan_to_num(density)
        # Smooth for nicer contours (optional)
        density_smooth = gaussian_filter(density, sigma=1.0)
        # Set levels for density
        max_density = density_smooth.max()
        min_density = max_density * 0.01
        levels = np.geomspace(min_density, max_density, 20)
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        cs = ax.contourf(X, Y, density_smooth.T, levels=levels, cmap=cmap, alpha=1.0, antialiased=True)
        # Add colorbar for each component
        # cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.01, shrink=0.7)
        # cbar.set_label(f'{label} count per dex$^2$')
        # Manually create a legend handle for this component
        color = plt.get_cmap(cmap)(0.7)
        legend_handles.append(Patch(facecolor=color, edgecolor='k', label=label, alpha=1.0))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0e10, 1.0e15)
    ax.set_ylim(1.0e-2, 1.0e14)
    ax.set_xlabel(r'Halo Mass [$M_{\odot}\ h^{-1}$]')
    ax.set_ylabel(r'IGM/ICS/CGM/Dust [$M_{\odot}\ h^{-1}$]')
    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()
    plt.savefig(OutputDir + 'Extragalactic_vs_HaloMass_contour' + OutputFormat)
    plt.close()

# -------------------------------------------------------

    # print('Plotting the spatial distribution of hot gas around galaxies')

    # plt.figure(figsize=(18, 5))  # New figure

    # w = np.where((Mvir > 0.0) & (StellarMass > 1.0e6) & (Type==0))[0]
    # if(len(w) > dilute): w = sample(list(w), dilute)

    # xx = Posx[w]
    # yy = Posy[w]
    # zz = Posz[w]

    # buff = BoxSize*0.1

    # logHotGas = np.log10(HotGas[w])
    # logICS = np.log10(IntraClusterStars[w])
    # logCGMgas = np.log10(CGMgas[w])

    # ax = plt.subplot(131)
    # sc1 = plt.scatter(xx, yy, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    # sc1_2 = plt.scatter(xx, yy, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    # plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    # plt.ylabel(r'$\mathrm{x}$')
    # plt.xlabel(r'$\mathrm{y}$')
    # plt.title('x vs y')

    # ax = plt.subplot(132)
    # sc2 = plt.scatter(xx, zz, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    # sc2_2 = plt.scatter(xx, zz, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    # plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    # plt.ylabel(r'$\mathrm{x}$')
    # plt.xlabel(r'$\mathrm{z}$')
    # plt.title('x vs z')


    # ax = plt.subplot(133)
    # sc3 = plt.scatter(yy, zz, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    # sc3_2 = plt.scatter(yy, zz, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    # plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    # plt.ylabel(r'$\mathrm{y}$')
    # plt.xlabel(r'$\mathrm{z}$')
    # plt.title('y vs z')

    # # Set face color to black for all 2D plots
    # for ax in plt.gcf().axes:
    #     ax.set_facecolor('black')
    #     ax.grid(False) # No grid
    #     # Set axis ticks to white
    #     ax.tick_params(axis='x', colors='k')
    #     ax.tick_params(axis='y', colors='k') 

    # # plt.tight_layout()
        

    # outputFile = OutputDir + 'SpatialDistribution' + OutputFormat
    # plt.savefig(outputFile)  # Save the figure
    # plt.close()

    # -------------------------------------------------------
    # Plotting ICS vs Redshift (full snapshot range)
    print('Plotting ICS vs Redshift')

    FirstSnap = 0          # First snapshot to read
    LastSnap = 63          # Last snapshot to read
    redshifts_arr = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073, 
                 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 
                 2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905, 
                 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 
                 0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

    ics_means = []
    ics_stds = []
    for snapnum in range(FirstSnap, LastSnap+1):
        snap_str = f'Snap_{snapnum}'
        try:
            ICS = read_hdf(snap_num=snap_str, param='IntraClusterStars') * 1.0e10 / Hubble_h
            Mvir = read_hdf(snap_num=snap_str, param='Mvir') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(snap_num=snap_str, param='StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(snap_num=snap_str, param='Type')
            w_cen = np.where((Mvir > 0.0) & (ICS > 0.0))[0]
            ICS_cen = ICS[w_cen]
            ICS_cen_frac = ICS_cen / (Mvir[w_cen] * 0.17) # Convert to fraction of total baryons in the halo (assuming cosmic baryon fraction of 0.17)
            if len(ICS_cen_frac) > 0:
                ics_means.append(np.mean(ICS_cen_frac))
                ics_stds.append(np.std(ICS_cen_frac))
            else:
                ics_means.append(np.nan)
                ics_stds.append(np.nan)
        except Exception as e:
            print(f"  Warning: Could not read snapshot {snap_str}: {e}")
            ics_means.append(np.nan)
            ics_stds.append(np.nan)

    redshifts_arr = np.array(redshifts_arr)
    ics_means = np.array(ics_means)
    ics_stds = np.array(ics_stds)

    plt.figure()
    ax = plt.subplot(111)
    ax.errorbar(redshifts_arr, ics_means, yerr=ics_stds, fmt='o-', color='blue', label='Mean ICS (centrals)')
    # ax.scatter(redshifts_arr, ICS_cen_frac, s=20, color='orange', alpha=0.5, label='ICS fraction (centrals)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Mean IntraClusterStars [$M_\odot\ h^{-1}$]')
    ax.set_title('ICS vs Redshift')
    # Flip x-axis so redshift increases to the right
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OutputDir + 'ICS_vs_Redshift' + OutputFormat)
    plt.close()

    # -------------------------------------------------------

    # # Plotting ICS vs redshift (full snapshot range) - individual galaxies

    # plt.figure()
    # ax = plt.subplot(111)

    # for snapnum in range(FirstSnap, LastSnap+1):
    #     snap_str = f'Snap_{snapnum}'
    #     try:
    #         ICS = read_hdf(snap_num=snap_str, param='IntraClusterStars') * 1.0e10 / Hubble_h
    #         Mvir = read_hdf(snap_num=snap_str, param='Mvir') * 1.0e10 / Hubble_h
    #         StellarMass = read_hdf(snap_num=snap_str, param='StellarMass') * 1.0e10 / Hubble_h
    #         Type = read_hdf(snap_num=snap_str, param='Type')
    #         w_cen = np.where((Mvir > 0.0) & (ICS > 0.0))[0]
    #         ICS_cen = ICS[w_cen]
    #         ICS_cen_frac = ICS_cen / (Mvir[w_cen] * 0.17) # Convert to fraction of total baryons in the halo (assuming cosmic baryon fraction of 0.17)
    #         ax.scatter([redshifts_arr[snapnum]]*len(ICS_cen_frac), ICS_cen_frac, s=20, alpha=0.5)
    #     except Exception as e:
    #         print(f"  Warning: Could not read snapshot {snap_str}: {e}")

    # ax.set_xlabel('Redshift')
    # ax.set_ylabel(r'IntraClusterStars / (Mvir * 0.17)')
    # ax.set_title('ICS fraction vs Redshift (centrals)')
    # # Flip x-axis so redshift increases to the right
    # ax.set_xlim(0, 0.7) 
    # ax.set_ylim(0, 0.5)

    # plt.tight_layout()
    # plt.savefig(OutputDir + 'ICS_fraction_vs_Redshift_individual' + OutputFormat)
    # plt.close()

    # -------------------------------------------------------

    # Plotting ICS vs Mvir at z=0

    plt.figure()
    ax = plt.subplot(111)

    w_z0 = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6) & (ICS > 0.0))[0]
    if len(w_z0) > dilute: w_z0 = sample(list(w_z0), dilute)
    Mvir_z0 = Mvir[w_z0]
    ICS_z0 = IntraClusterStars[w_z0]
    ICS_z0_frac_z0 = ICS_z0 / (Mvir_z0 * 0.17) # Convert to fraction of total baryons in the halo (assuming cosmic baryon fraction of 0.17)

    ax.scatter(Mvir_z0, ICS_z0_frac_z0, s=20, alpha=0.5)
    ax.set_xlabel(r'Mvir [$M_\odot\ h^{-1}$]')
    ax.set_ylabel(r'IntraClusterStars / (Mvir * 0.17)')
    ax.set_title('ICS vs Mvir at z=0')
    ax.set_xscale('log')
    # ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OutputDir + 'ICS_vs_Mvir_at_z0' + OutputFormat)
    plt.close()

    # -------------------------------------------------------
    # print('Plotting 3D spatial distribution with box and black background')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # w_box = np.where((Mvir > 0.0) & (StellarMass > 1.0e8) & (Type==0))[0]
    # if(len(w_box) > dilute): w_box = sample(list(w_box), dilute)

    # xx_box = Posx[w_box]
    # yy_box = Posy[w_box]
    # zz_box = Posz[w_box]
    # logHotGas_box = np.log10(HotGas[w_box])
    # logICS_box = np.log10(IntraClusterStars[w_box])

    # # Plot Hot Gas (PuRd, faint, larger points)
    # ax.scatter(xx_box, yy_box, zz_box, s=20, c=logHotGas_box, cmap='PuRd', alpha=0.07, edgecolors='none', label='Hot Gas')
    # # Plot ICS (Blues, small, brighter points)
    # ax.scatter(xx_box, yy_box, zz_box, s=0.2, c=logICS_box, cmap='Blues', alpha=0.8, edgecolors='none', label='ICS')

    # # Draw the box edges
    # points = np.array([[0,0,0], [BoxSize,0,0], [BoxSize,BoxSize,0], [0,BoxSize,0],
    #                    [0,0,BoxSize], [BoxSize,0,BoxSize], [BoxSize,BoxSize,BoxSize], [0,BoxSize,BoxSize]])

    # edges = [[points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],
    #          [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],
    #          [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]]
    # line_collection = Line3DCollection(edges, colors='k')
    # ax.add_collection3d(line_collection)

    # ax.set_xlabel('X (Mpc/h)')
    # ax.set_ylabel('Y (Mpc/h)')
    # ax.set_zlabel('Z (Mpc/h)')

    # ax.set_xlim([0, BoxSize])
    # ax.set_ylim([0, BoxSize])
    # ax.set_zlim([0, BoxSize])

    # # Set background color to black for the 3D plot
    # ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    # ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    # ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    # ax.grid(False)
    # ax.tick_params(axis='x', colors='k')
    # ax.tick_params(axis='y', colors='k')
    # ax.tick_params(axis='z', colors='k')

    # ax.set_box_aspect([1,1,1])

    # plt.tight_layout()

    # outputFile_box = OutputDir + 'SpatialDistribution3D_Box' + OutputFormat
    # plt.savefig(outputFile_box)
    # print('Saved file to', outputFile_box, '\n')
    # plt.close()

    # -------------------------------------------------------

    print('Plotting extra-galactic mass/gas + dust vs temperature')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    Mvir_cen = Mvir[w_cen]
    HotGas_cen = HotGas[w_cen]
    ICS_cen = IntraClusterStars[w_cen]
    CGMgas_cen = CGMgas[w_cen]
    Dust_cen = HotGasDust[w_cen] + CGMgasDust[w_cen] + EjectedMassDust[w_cen]
    Temp_cen = Tvir[w_cen]

    HotGas_frac = HotGas_cen / Mvir_cen
    ICS_frac = ICS_cen / Mvir_cen
    CGMgas_frac = CGMgas_cen / Mvir_cen
    Dust_frac = Dust_cen / Mvir_cen

    # Define bins (logarithmic for temperature, linear for fractions)
    x_bins = np.logspace(4, 8, 250)
    y_bins = np.linspace(0, 1, 250)

    # List of (component, label, colormap)
    components = [
        # (CGMgas_frac, 'CGM', 'Greens_r'),
        # (Dust_frac, 'Dust', 'Blues_r'),
        (HotGas_frac, 'IGM/ICM', 'PuRd_r'),
        # (ICS_frac, 'ICS', 'plasma')
    ]

    from scipy.ndimage import gaussian_filter


    # Plot filled contours and collect legend handles manually
    from matplotlib.patches import Patch
    legend_handles = []
    for comp, label, cmap in components:
        # 2D histogram for number count
        H, xedges, yedges = np.histogram2d(Temp_cen, comp, bins=[x_bins, y_bins])
        # Convert to number density per dex^2
        # Bin widths in log10 space
        dx = np.diff(np.log10(xedges))
        dy = np.diff(yedges)
        bin_area = dx[:, None] * dy[None, :]  # shape (nx, ny)
        # Avoid division by zero
        bin_area[bin_area == 0] = np.nan
        density = H / bin_area  # number per dex^2
        density = np.nan_to_num(density)
        # Smooth for nicer contours (optional)
        density_smooth = gaussian_filter(density, sigma=1.0)
        # Set levels for density
        max_density = density_smooth.max()
        min_density = max_density * 0.01
        levels = np.geomspace(min_density, max_density, 20)
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        cs = ax.contourf(X, Y, density_smooth.T, levels=levels, cmap=cmap, alpha=1.0, antialiased=True)
        # Add colorbar for each component
        # cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.01, shrink=0.7)
        # cbar.set_label(f'{label} count per dex$^2$')
        # Manually create a legend handle for this component
        color = plt.get_cmap(cmap)(0.7)
        legend_handles.append(Patch(facecolor=color, edgecolor='k', label=label, alpha=1.0))

    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlim(1.0e4, 1.0e8)
    ax.set_ylim(0, 0.08)
    ax.set_xlabel(r'Temperature [K]')
    ax.set_ylabel(r'IGM/ICS/CGM/Dust [$M_{\odot}\ h^{-1}$]')
    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()
    plt.savefig(OutputDir + 'Extragalactic_vs_Temperature_contour' + OutputFormat)
    plt.close()

    # -------------------------------------------------------

    # # Plotting surface density components vs disc radius (for the most massive central galaxy)

    # plt.figure()  # New figure
    # ax = plt.subplot(111)  # 1 plot on the figure

    # w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    # AnnuliiRadius_cen = AnnuliiRadius[w_cen] * 1.0e3 / Hubble_h  # Convert from Mpc/h to kpc
    # Disc_gas_cen = Disc_gas[w_cen]
    # Disc_stars_cen = Disc_stars[w_cen]
    # Disc_sfr_cen = Disc_sfr[w_cen]
    # Disc_h2_cen = Disc_h2[w_cen]
    # Disc_dust_cen = Disc_dust[w_cen]
    # Disc_gasmetals_cen = Disc_GasMetals[w_cen]
    # Disc_starsmetals_cen = Disc_StarsMetals[w_cen]

    # # Get stellar mass, Mvir, Rvir, Vvir for centrals
    # StellarMass_cen = StellarMass[w_cen]
    # Mvir_cen = Mvir[w_cen]
    # Rvir_cen = Rvir[w_cen] * 1.0e3 / Hubble_h  # Convert from Mpc/h to kpc
    # Vvir_cen = Vvir[w_cen]  # km/s

    # # Select a few galaxies at different masses for diagnostic output
    # sorted_idx = np.argsort(StellarMass_cen)[::-1]  # Sort by stellar mass descending
    # n_gals = len(sorted_idx)
    # diagnostic_indices = [
    #     sorted_idx[0],                    # Most massive
    #     sorted_idx[n_gals // 4],          # 25th percentile
    #     sorted_idx[n_gals // 2],          # Median
    #     sorted_idx[3 * n_gals // 4],      # 75th percentile
    #     sorted_idx[-1],                   # Least massive
    # ]

    # print("\n" + "="*80)
    # print("DIAGNOSTIC: Checking disc bins for galaxies at different masses")
    # print("="*80)

    # for j, gal_idx in enumerate(diagnostic_indices):
    #     radii_diag = AnnuliiRadius_cen[gal_idx]
    #     gas_profile_diag = Disc_gas_cen[gal_idx]
    #     stars_profile_diag = Disc_stars_cen[gal_idx]

    #     # Compute annulus areas in kpc^2
    #     annulus_areas_diag = np.pi * (radii_diag[1:]**2 - radii_diag[:-1]**2)

    #     # Find last physical bin (where r_out > r_in)
    #     last_physical = 0
    #     for i in range(len(radii_diag)-1):
    #         if radii_diag[i+1] > radii_diag[i]:
    #             last_physical = i

    #     print(f"\nGalaxy {j+1}: M*={StellarMass_cen[gal_idx]:.2e} Msun/h, Mvir={Mvir_cen[gal_idx]:.2e} Msun/h")
    #     print(f"Halo Rvir={Rvir_cen[gal_idx]:.2f} kpc, Vvir={Vvir_cen[gal_idx]:.1f} km/s")
    #     print(f"Last physical bin: {last_physical}, Disc outer radius={radii_diag[last_physical+1]:.2f} kpc")
    #     print("-"*70)

    #     # Print bins around the transition (last few physical + first few zero-area)
    #     start_bin = max(0, last_physical - 3)
    #     end_bin = min(len(gas_profile_diag), last_physical + 5)

    #     for i in range(start_bin, end_bin):
    #         area_str = f"{annulus_areas_diag[i]:.2e}" if annulus_areas_diag[i] > 0 else "0.00e+00"
    #         physical = "PHYSICAL" if i <= last_physical else "ZERO-AREA"
    #         print(f"Bin {i:2d}: r_in={radii_diag[i]:6.2f}, r_out={radii_diag[i+1]:6.2f}, "
    #               f"Gas={gas_profile_diag[i]:9.2e}, Stars={stars_profile_diag[i]:9.2e}, "
    #               f"Area={area_str} [{physical}]")

    # print("\n" + "="*80 + "\n")

    # # Full bin diagnostics for 3 representative galaxies
    # print("\n" + "="*100)
    # print("FULL BIN DIAGNOSTICS: Low-mass, Medium-mass, and High-mass galaxies")
    # print("="*100)

    # # Select 3 galaxies: high-mass (top 5%), medium-mass (median), low-mass (bottom 5%)
    # full_diag_indices = [
    #     sorted_idx[0],                    # Most massive (high-mass)
    #     sorted_idx[n_gals // 2],          # Median (medium-mass)
    #     sorted_idx[int(n_gals * 0.95)],   # 95th percentile (low-mass)
    # ]
    # mass_labels = ["HIGH-MASS", "MEDIUM-MASS", "LOW-MASS"]

    # for j, gal_idx_full in enumerate(full_diag_indices):
    #     radii_full = AnnuliiRadius_cen[gal_idx_full]
    #     gas_full = Disc_gas_cen[gal_idx_full]
    #     stars_full = Disc_stars_cen[gal_idx_full]
    #     dust_full = Disc_dust_cen[gal_idx_full]
    #     h2_full = Disc_h2_cen[gal_idx_full]
    #     sfr_full = Disc_sfr_cen[gal_idx_full]

    #     # Compute annulus areas in kpc^2
    #     areas_full = np.pi * (radii_full[1:]**2 - radii_full[:-1]**2)

    #     # Find last physical bin
    #     last_phys = 0
    #     for i in range(len(radii_full)-1):
    #         if radii_full[i+1] > radii_full[i]:
    #             last_phys = i

    #     print(f"\n{'='*100}")
    #     print(f"{mass_labels[j]} GALAXY: M*={StellarMass_cen[gal_idx_full]:.2e} Msun/h, "
    #           f"Mvir={Mvir_cen[gal_idx_full]:.2e} Msun/h")
    #     print(f"Halo Rvir={Rvir_cen[gal_idx_full]:.2f} kpc, Vvir={Vvir_cen[gal_idx_full]:.1f} km/s, "
    #           f"Disc outer radius={radii_full[last_phys+1]:.2f} kpc")
    #     print(f"{'='*100}")
    #     print(f"{'Bin':>4} {'r_in':>8} {'r_out':>8} {'Area':>10} {'Gas':>12} {'Stars':>12} "
    #           f"{'Dust':>10} {'H2':>10} {'SFR':>10} {'Status':<10}")
    #     print("-"*100)

    #     for i in range(len(gas_full)):
    #         area_val = areas_full[i]
    #         status = "PHYSICAL" if i <= last_phys else "ZERO-AREA"

    #         # Format numbers, handling very small values
    #         def fmt(val):
    #             if val == 0:
    #                 return "0.00"
    #             elif abs(val) < 1e-10:
    #                 return f"{val:.1e}"
    #             elif abs(val) < 1e-2:
    #                 return f"{val:.2e}"
    #             elif abs(val) > 1e6:
    #                 return f"{val:.2e}"
    #             else:
    #                 return f"{val:.2f}"

    #         print(f"{i:4d} {radii_full[i]:8.2f} {radii_full[i+1]:8.2f} {fmt(area_val):>10} "
    #               f"{fmt(gas_full[i]):>12} {fmt(stars_full[i]):>12} "
    #               f"{fmt(dust_full[i]):>10} {fmt(h2_full[i]):>10} {fmt(sfr_full[i]):>10} {status:<10}")

    #     # Summary stats
    #     total_gas = np.sum(gas_full)
    #     total_stars = np.sum(stars_full)
    #     total_dust = np.sum(dust_full)
    #     print("-"*100)
    #     print(f"TOTALS: Gas={total_gas:.2e}, Stars={total_stars:.2e}, Dust={total_dust:.2e}")

    # print("\n" + "="*100 + "\n")

    # # # Select the most massive central galaxy for the plot
    # # gal_idx = np.argmax(StellarMass_cen)

    # # For the plot, select the median galaxy to avoid extreme outliers
    # # gal_idx = sorted_idx[n_gals // 2]

    # Galaxy_masses = 1.0e9  # Msun/h
    # sfr_max = 1.0e5  # Msun/yr

    # # For the plot find a glaxy with Galaxy_masses close to 1e10 Msun/h (if it exists), otherwise take the median
    # gal_idx = None
    # for idx in sorted_idx:
    #     if abs(StellarMass_cen[idx] - Galaxy_masses) / Galaxy_masses < 0.1 and Disc_sfr_cen[idx][-1] < sfr_max:  # Within 10%
    #         gal_idx = idx
    #         break
    # if gal_idx is None:
    #     gal_idx = sorted_idx[n_gals // 2]  # Fall back to median if no close match

    # # For the plot, select a galaxy with a well-defined disc (e.g. where the last physical bin is reasonably large)
    # # gal_idx = None
    # # for idx in sorted_idx:
    # #     radii_check = AnnuliiRadius_cen[idx]
    # #     last_physical_check = 0
    # #     for i in range(len(radii_check)-1):
    # #         if radii_check[i+1] > radii_check[i]:
    # #             last_physical_check = i
    # #     if last_physical_check >= 5:  # Require at least 5 physical bins
    # #         gal_idx = idx
    # #         break

    # # For the plot, select a galaxy with Galaxy_masses and SFR > sfr_min
    # # gal_idx = None
    # # for idx in sorted_idx:
    # #     if abs(StellarMass_cen[idx] - Galaxy_masses) / Galaxy_masses < 0.5 and Disc_sfr_cen[idx][-1] > sfr_min:
    # #         gal_idx = idx
    # #         break
    # # if gal_idx is None:
    # #     gal_idx = sorted_idx[n_gals // 2]  # Fall back to median if no close match

    # print(f"Selected galaxy for surface density plot: M*={StellarMass_cen[gal_idx]:.2e} Msun/h, "
    #       f"Mvir={Mvir_cen[gal_idx]:.2e} Msun/h, Rvir={Rvir_cen[gal_idx]:.2f} kpc, Disc outer radius={AnnuliiRadius_cen[gal_idx][-1]:.2f} kpc")
    
    # print(f"\nNumber of galaxies with Disc outer radius > Rvir: {np.sum(AnnuliiRadius_cen[:, -1] > Rvir_cen)} out of {len(StellarMass_cen)}")
    # print(f"\n Sample of these galaxies:")
    # for i in range(min(10, len(StellarMass_cen))):
    #     if AnnuliiRadius_cen[i, -1] > Rvir_cen[i]:
    #         print(f"Galaxy {i}: M*={StellarMass_cen[i]:.2e} Msun/h, Mvir={Mvir_cen[i]:.2e} Msun/h, Rvir={Rvir_cen[i]:.2f} kpc, Disc outer radius={AnnuliiRadius_cen[i][-1]:.2f} kpc")

    # radii = AnnuliiRadius_cen[gal_idx]  # shape: (num_annuli+1,)
    # gas_profile = Disc_gas_cen[gal_idx]  # shape: (num_annuli,)
    # stars_profile = Disc_stars_cen[gal_idx]
    # sfr_profile = Disc_sfr_cen[gal_idx]
    # h2_profile = Disc_h2_cen[gal_idx]
    # dust_profile = Disc_dust_cen[gal_idx]
    # gasmetals_profile = Disc_gasmetals_cen[gal_idx]
    # starsmetals_profile = Disc_starsmetals_cen[gal_idx]

    # # Compute annulus areas in kpc^2
    # annulus_areas = np.pi * (radii[1:]**2 - radii[:-1]**2)

    # # Compute surface densities
    # gas_sd = gas_profile / annulus_areas
    # stars_sd = stars_profile / annulus_areas
    # sfr_sd = sfr_profile / annulus_areas
    # h2_sd = h2_profile / annulus_areas
    # dust_sd = dust_profile / annulus_areas
    # gasmetals_sd = gasmetals_profile / annulus_areas
    # starsmetals_sd = starsmetals_profile / annulus_areas

    # plt.step(radii[:-1], gas_sd, where='post', label='Disc Gas', color='blue')
    # plt.step(radii[:-1], stars_sd, where='post', label='Disc Stars', color='orange')
    # plt.step(radii[:-1], sfr_sd, where='post', label='Disc SFR', color='green')
    # plt.step(radii[:-1], h2_sd, where='post', label='Disc H2', color='purple')
    # plt.step(radii[:-1], dust_sd, where='post', label='Disc Dust', color='brown')
    # plt.step(radii[:-1], gasmetals_sd, where='post', label='Disc Gas Metals', color='red')
    # plt.step(radii[:-1], starsmetals_sd, where='post', label='Disc Stars Metals', color='cyan')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'Radius [kpc]')
    # plt.ylabel(r'Surface Density [$h^{-1} M_{\odot}$ kpc$^{-2}$ / $h^{-1} M_{\odot}$ yr$^{-1}$ kpc$^{-2}$]')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OutputDir + 'DiscSurfaceDensity_vs_Radius' + OutputFormat)
    # plt.close()

    # -------------------------------------------------------

    # Plotting mass functions of the different components (hot gas, ICS, CGM, dust)

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    HotGas_cen = HotGas[w_cen]
    ICS_cen = IntraClusterStars[w_cen]
    CGMgas_cen = CGMgas[w_cen]
    Dust_cen = HotGasDust[w_cen] + CGMgasDust[w_cen] + EjectedMassDust[w_cen]

    components = [
        (CGMgas_cen, 'CGM', 'green'),
        (ICS_cen, 'ICS', 'orange'),
        (Dust_cen, 'Dust', 'blue'),
        (HotGas_cen, 'IGM/ICM', 'red')
    ]

    binwidth = 0.25  # mass function histogram bin width in log10 space

    for comp, label, color in components:
        mi = np.log10(comp[comp > 0.0].min()) - 0.5
        ma = np.log10(comp.max()) + 0.5
        NB = int((ma - mi) / binwidth)
        counts, binedges = np.histogram(np.log10(comp), range=(mi, ma), bins=NB)
        bin_centers = 0.5 * (binedges[:-1] + binedges[1:])
        xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins
        plt.plot(xaxeshisto, np.log10(counts/volume/binwidth), label=label, color=color)

    plt.axis([6, 14, -6, -1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.ylabel(r'$\log_{10}\ \phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    plt.xlabel(r'$\log_{10} M_{\mathrm{gas/mass}}\ (M_{\odot})$')

    plt.legend()
    plt.tight_layout()
    plt.savefig(OutputDir + 'MassFunction_ExtragalacticComponents' + OutputFormat)
    plt.close()

    # -------------------------------------------------------

    # Plotting surface density profiles of the different components (hot gas, ICS, CGM, dust) for 10 haloes as a function of halo mass

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    if len(w_cen) > dilute: w_cen = sample(list(w_cen), dilute)
    Mvir_cen = Mvir[w_cen]
    HotGas_cen = HotGas[w_cen]
    ICS_cen = IntraClusterStars[w_cen]
    CGMgas_cen = CGMgas[w_cen]
    Dust_cen = HotGasDust[w_cen] + CGMgasDust[w_cen] + EjectedMassDust[w_cen]
    HotGasDust_cen = HotGasDust[w_cen]
    CGMgasDust_cen = CGMgasDust[w_cen]
    EjectedMassDust_cen = EjectedMassDust[w_cen]
    Rvir_cen = Rvir[w_cen]
    Rvir_cen_kpc = Rvir_cen * 1.0e3 / Hubble_h  # Convert from Mpc/h to kpc

    area_kpc = 2 * np.pi * (Rvir_cen_kpc**2)  # Convert from Mpc^2/h^2 to kpc^2

    components = [
        (CGMgas_cen / area_kpc, 'CGM', 'green'),
        (ICS_cen / area_kpc, 'ICS', 'orange'),
        # (Dust_cen / area_kpc, 'Dust', 'blue'),
        (HotGas_cen / area_kpc, 'IGM/ICM', 'red'),
        # (Mvir_cen / area_kpc, 'Halo Mass', 'black')
        (HotGasDust_cen / area_kpc, 'Hot Gas Dust', 'purple'),
        (CGMgasDust_cen / area_kpc, 'CGM Dust', 'brown'),
        (EjectedMassDust_cen / area_kpc, 'Ejected Dust', 'cyan')
    ]

    for comp, label, color in components:
        plt.plot(Rvir_cen_kpc, comp, marker='.', linestyle='', markersize=5, alpha=0.5, label=label, color=color)

    # plt.xscale('log')
    plt.yscale('log')
    # plt.axis([1.0e10, 1.0e15, 1.0e-3, 1.0e6])
    plt.xlabel(r'Halo Radius [$M_{\odot}\ h^{-1}$]')
    plt.ylabel(r'Surface Density [$M_{\odot}$ kpc$^{-2}$]')
    plt.legend()
    plt.tight_layout()

    plt.savefig(OutputDir + 'SurfaceDensity_vs_HaloRadius' + OutputFormat)
    plt.close()

    # -------------------------------------------------------

    import fsps
    import mpi4py.MPI as MPI

    # -------------------------------------------------------
    # 1. Initialize MPI
    # -------------------------------------------------------

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # We load the HDF5 file on all ranks.
    sfr_history = read_hdf(snap_num=f'Snap_{LastSnap}', param='SfrHistory')
    DiscDust = read_hdf(snap_num=f'Snap_{LastSnap}', param='DiscDust') * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(snap_num=f'Snap_{LastSnap}', param='DiskRadius') # Ensure this is in Mpc/h or kpc 

    # Rank 0 defines the workload and splits it
    if rank == 0:
        w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e8) & (sfr_history.any(axis=1) > 0.0))[0]
        # Bumping up to 500 so you can clearly see the distributions in the new plots
        w_cen = w_cen[:500]  
        print(f"Total central galaxies to process: {len(w_cen)}")
        
        chunks = np.array_split(w_cen, size)
    else:
        chunks = None

    local_w_cen = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"Distributing across {size} MPI ranks...")

    # -------------------------------------------------------
    # 2. Initialize FSPS locally on EACH rank
    # -------------------------------------------------------
    # Set zcontinuous=1 for the single final-metallicity approximation
    sp = fsps.StellarPopulation(sfh=3, zcontinuous=3)

    num_timebins = sfr_history.shape[1]
    time_bins_gyr = np.linspace(0.1, 13.7, num_timebins) 

    # Arrays to store the output for THIS rank
    local_g_mags = np.zeros(len(local_w_cen))
    local_r_mags = np.zeros(len(local_w_cen))
    local_masses = np.zeros(len(local_w_cen)) 
    local_ssfrs  = np.zeros(len(local_w_cen)) 

    # -------------------------------------------------------
    # 3. The Computation Loop
    # -------------------------------------------------------
    # Set a seed so your random inclinations are reproducible across runs!
    np.random.seed(42 + rank) 
    
    for idx, gal_index in enumerate(local_w_cen):
        
        gal_sfr = sfr_history[gal_index, :]
        
        # Calculate final metallicity fraction Z
        gal_Z_scalar = MetalsStellarMass[gal_index] / (StellarMass[gal_index] + 1e-9)
        gal_Z_scalar = np.clip(gal_Z_scalar, 1e-4, 0.05) 
        gal_Z_array = np.full_like(gal_sfr, gal_Z_scalar)
        
        sp.set_tabular_sfh(time_bins_gyr, gal_sfr, Z=gal_Z_array)

        # --- THE DUST & INCLINATION LOGIC ---
        # 1. Total disk dust mass
        gal_dust_mass = np.sum(DiscDust[gal_index]) 
        
        # 2. Convert radius to parsecs (Assuming DiskRadius is in Mpc/h)
        gal_radius_pc = (DiskRadius[gal_index] / 0.73) * 1.0e6 
        
        # 3. Calculate Face-On Surface Density (M_sun / pc^2)
        if gal_radius_pc > 0 and gal_dust_mass > 0:
            face_on_surface_density = gal_dust_mass / (np.pi * gal_radius_pc**2)
        else:
            face_on_surface_density = 0.0
            
        # 4. Apply Random Inclination 
        # Draw cos(theta) uniformly between 0.1 (edge-on, avoiding div by 0) and 1.0 (face-on)
        cos_theta = np.random.uniform(0.1, 1.0)
        inclined_surface_density = face_on_surface_density / cos_theta
            
        # 5. Calculate Optical Depths
        gamma_dust = 0.06  # Mass-extinction coefficient (tuneable)
        mu = 0.3           # Fraction of attenuation in the diffuse ISM
        
        dust2_val = gamma_dust * inclined_surface_density
        # THE FIX: Apply a physical ceiling to the optical depth.
        # An optical depth of 3.0 means ~95% of the light is blocked. 
        # Anything beyond this should act as an opaque wall, not an infinite reddener.
        dust2_val = np.clip(dust2_val, 0.0, 3.0) 
        
        dust1_val = dust2_val * (1.0 / mu - 1.0)
        
        # 6. Apply to FSPS
        sp.params['dust_type'] = 0  # 0 = Charlot & Fall (2000)
        sp.params['dust1'] = dust1_val
        sp.params['dust2'] = dust2_val
        # ------------------------------------
        
        mags = sp.get_mags(tage=13.7, bands=['sdss_g', 'sdss_r'])
        
        local_g_mags[idx] = mags[0]
        local_r_mags[idx] = mags[1]
        
        # Track Mass and sSFR for plotting
        local_masses[idx] = StellarMass[gal_index]
        total_sfr = SfrDisk[gal_index] + SfrBulge[gal_index]
        local_ssfrs[idx] = np.log10(total_sfr / StellarMass[gal_index] + 1e-14)

    # -------------------------------------------------------
    # 4. Gather the Results back to Rank 0
    # -------------------------------------------------------
    gathered_g = comm.gather(local_g_mags, root=0)
    gathered_r = comm.gather(local_r_mags, root=0)
    gathered_m = comm.gather(local_masses, root=0)
    gathered_ssfr = comm.gather(local_ssfrs, root=0)

    # -------------------------------------------------------
    # 5. Plotting on Rank 0
    # -------------------------------------------------------
    if rank == 0:
        final_g_mags = np.concatenate(gathered_g)
        final_r_mags = np.concatenate(gathered_r)
        final_masses = np.concatenate(gathered_m)
        final_ssfrs  = np.concatenate(gathered_ssfr)
        
        color_g_r = final_g_mags - final_r_mags
        
        # Separate populations based on your sSFRcut (-11.0)
        quiescent = final_ssfrs < sSFRcut
        starforming = final_ssfrs >= sSFRcut
        
        print(f"Calculation complete.")
        print(f"Quiescent: {np.sum(quiescent)}, Star-Forming: {np.sum(starforming)}")

        # --- PLOT 1: g-r Color-Magnitude Diagram ---
        plt.figure(figsize=(8, 6))
        plt.scatter(final_r_mags[quiescent], color_g_r[quiescent], 
                    c='crimson', alpha=0.6, label='Quiescent', edgecolors='none')
        plt.scatter(final_r_mags[starforming], color_g_r[starforming], 
                    c='dodgerblue', alpha=0.6, label='Star-Forming', edgecolors='none')
        
        plt.xlabel('Absolute r-band Magnitude ($M_r$)')
        plt.ylabel('Color ($g - r$)')
        plt.title('Color-Magnitude Diagram for Central Galaxies')
        plt.gca().invert_xaxis()  
        plt.legend()
        plt.tight_layout()
        plt.savefig(OutputDir + 'CentralGalaxies_CMD_g_vs_gr' + OutputFormat)
        plt.close()

        # --- PLOT 2: g-r Color vs Stellar Mass ---
        plt.figure(figsize=(8, 6))
        plt.scatter(np.log10(final_masses[quiescent]), color_g_r[quiescent], 
                    c='crimson', alpha=0.6, label='Quiescent', edgecolors='none')
        plt.scatter(np.log10(final_masses[starforming]), color_g_r[starforming], 
                    c='dodgerblue', alpha=0.6, label='Star-Forming', edgecolors='none')
        
        plt.xlabel(r'$\log_{10}$ Stellar Mass [$M_{\odot}\ h^{-1}$]')
        plt.ylabel('Color ($g - r$)')
        plt.title('Color vs. Stellar Mass')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OutputDir + 'CentralGalaxies_Color_vs_Mass' + OutputFormat)
        plt.close()

    # --- PLOT 3: The r-band Luminosity Function ---
        plt.figure(figsize=(8, 6))
        
        # 1. Define the magnitude bins (from faint to bright)
        dM = 0.5
        mag_bins = np.arange(-15.0, -25.0, -dM) # Note: going backwards because magnitudes are inverted
        mag_bins = np.sort(mag_bins) # Sort them to be strictly increasing for the histogram
        
        # 2. Calculate the histogram (number of galaxies in each bin)
        # We use all galaxies here (quiescent + star-forming)
        counts, bin_edges = np.histogram(final_r_mags, bins=mag_bins)
        
        # 3. Calculate the volume density (Phi)
        # phi = N / (Volume * bin_width)
        # Volume was calculated at the top of your script: volume = (BoxSize/Hubble_h)**3.0 * VolumeFraction
        phi = counts / (volume * dM)
        
        # 4. Find the center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Filter out empty bins to avoid log10(0) errors
        valid = phi > 0
        
        # Plot the SAM's Luminosity Function
        plt.plot(bin_centers[valid], np.log10(phi[valid]), 
                 marker='o', linestyle='-', color='black', linewidth=2, label='DarkSAGE SAM')

        # --- Observational Data Overlay (SDSS r-band, Blanton+2003) ---
        # Assuming your magnitudes are absolute (M_r) and your volume is in (Mpc/h)^3
        # Adjust M_star if your SAM outputs magnitudes strictly without the 'h' scaling
        M_star = -20.44 + 5 * np.log10(Hubble_h) 
        phi_star = 0.0149 # h^3 Mpc^-3
        alpha = -1.05
        
        # Calculate the theoretical Schechter curve
        M_plot = np.linspace(-15, -24, 100)
        M_diff = 0.4 * (M_star - M_plot)
        schechter_phi = 0.4 * np.log(10) * phi_star * (10 ** M_diff) ** (alpha + 1) * np.exp(-10 ** M_diff)
        
        # Overlay the observational curve
        plt.plot(M_plot, np.log10(schechter_phi), 
                 linestyle='--', color='red', linewidth=2, label='SDSS Observations (Blanton+2003)')
        # --------------------------------------------------------------

        plt.xlabel('Absolute r-band Magnitude ($M_r$)')
        plt.ylabel(r'$\log_{10}\ \Phi\ [h^3\ \mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$')
        plt.title('$r$-band Luminosity Function')
        plt.gca().invert_xaxis() # Standard convention: bright magnitudes on the left
        
        # Set realistic y-axis limits for an LF
        plt.ylim(-6, -1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OutputDir + 'CentralGalaxies_LuminosityFunction_r' + OutputFormat)
        plt.close()

    # --- PLOT 4: X-ray Luminosity Proxy vs. Halo Mass ---
        plt.figure(figsize=(8, 6))

        # Filter for group and cluster-sized central halos (Mvir > 10^12.5 M_sun)
        w_xray = np.where((Type == 0) & (Mvir > 10**12.5) & (Rvir > 0.0) & (HotGas > 0.0))[0]
        
        Mvir_xray = Mvir[w_xray]
        HotGas_xray = HotGas[w_xray]
        Tvir_xray = Tvir[w_xray]
        Rvir_xray = Rvir[w_xray] 
        
        # Calculate the Bremsstrahlung L_X proxy
        # Note: This is an unnormalized proxy, so we plot it in arbitrary log units
        Lx_proxy = (HotGas_xray**2 * np.sqrt(Tvir_xray)) / (Rvir_xray**3)
        
        # Logarithmic values for plotting
        log_Mvir = np.log10(Mvir_xray)
        log_Lx = np.log10(Lx_proxy)
        log_Tvir = np.log10(Tvir_xray)

        # Plot the scatter, color-coded by the Virial Temperature
        scatter = plt.scatter(log_Mvir, log_Lx, c=log_Tvir, cmap='plasma', 
                              alpha=0.6, edgecolors='none', s=30)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$\log_{10}$ Virial Temperature [K]')

        # --- Reference Slopes ---
        # Pick a normalization point near the massive end (e.g., Mvir = 10^14.5)
        # We will anchor the reference lines here to compare slopes
        anchor_mask = (log_Mvir > 14.4) & (log_Mvir < 14.6)
        if np.any(anchor_mask):
            M_anchor = 14.5
            Lx_anchor = np.median(log_Lx[anchor_mask])
            
            M_range = np.array([12.5, 15.0])
            
            # Pure Gravity (Self-Similar) Slope: 4/3 (~1.33)
            Lx_self_similar = Lx_anchor + (4/3) * (M_range - M_anchor)
            
            # Observed Slope with AGN Feedback (e.g., Pratt et al. 2009): ~1.65
            Lx_observed = Lx_anchor + 1.65 * (M_range - M_anchor)

            plt.plot(M_range, Lx_self_similar, 'k--', linewidth=2, label=r'Self-Similar ($M^{4/3}$)')
            plt.plot(M_range, Lx_observed, 'r-', linewidth=2, label=r'Observed + AGN Feedback ($M^{1.65}$)')

        plt.xlabel(r'$\log_{10}$ Halo Mass [$M_{\odot}\ h^{-1}$]')
        plt.ylabel(r'$\log_{10}$ $L_X$ Proxy [Arbitrary Units]')
        plt.title('Hot Gas X-ray Scaling Relation')
        
        # plt.legend()
        plt.tight_layout()
        plt.savefig(OutputDir + 'CentralGalaxies_Lx_vs_Mvir' + OutputFormat)
        plt.close()
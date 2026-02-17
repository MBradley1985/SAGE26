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

# -------------------------------------------------------------------------
    print('Plotting hot gas mass vs halo mass')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure   

    w_cen = np.where((Type==0) & (Mvir > 0.0) & (StellarMass > 1.0e6))[0]
    Mvir_cen = Mvir[w_cen]
    HotGas_cen = HotGas[w_cen]
    ICS_cen = IntraClusterStars[w_cen]
    CGMgas_cen = CGMgas[w_cen]

    plt.scatter(Mvir_cen, HotGas_cen, s=1, alpha=0.5, color='firebrick', label='Model')
    plt.scatter(Mvir_cen, ICS_cen, s=1, alpha=0.5, color='orange', label='Model - ICS')

    plt.xscale('log')
    plt.yscale('log')
    plt.axis([1.0e10, 1.0e15, 1.0e9, 1.0e14])
    plt.xlabel(r'Halo Mass [$h^{-1} M_{\odot}$]')
    plt.ylabel(r'IGM/ICS [$h^{-1} M_{\odot}$]')
    plt.tight_layout()
    plt.savefig(OutputDir + 'HotGas_vs_HaloMass' + OutputFormat)
    plt.close()

# -------------------------------------------------------

    print('Plotting the spatial distribution of hot gas around galaxies')

    plt.figure(figsize=(18, 5))  # New figure

    w = np.where((Mvir > 0.0) & (StellarMass > 1.0e6) & (Type==0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    xx = Posx[w]
    yy = Posy[w]
    zz = Posz[w]

    buff = BoxSize*0.1

    logHotGas = np.log10(HotGas[w])
    logICS = np.log10(IntraClusterStars[w])
    logCGMgas = np.log10(CGMgas[w])

    ax = plt.subplot(131)
    sc1 = plt.scatter(xx, yy, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    sc1_2 = plt.scatter(xx, yy, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')
    plt.xlabel(r'$\mathrm{y}$')
    plt.title('x vs y')

    ax = plt.subplot(132)
    sc2 = plt.scatter(xx, zz, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    sc2_2 = plt.scatter(xx, zz, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')
    plt.xlabel(r'$\mathrm{z}$')
    plt.title('x vs z')


    ax = plt.subplot(133)
    sc3 = plt.scatter(yy, zz, marker='o', s=20, c=logHotGas, cmap='PuRd', alpha=0.07, edgecolors='none')
    sc3_2 = plt.scatter(yy, zz, marker='.', s=0.5, c=logICS, cmap='Blues', alpha=0.8, edgecolors='none')
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{y}$')
    plt.xlabel(r'$\mathrm{z}$')
    plt.title('y vs z')

    # Set face color to black for all 2D plots
    for ax in plt.gcf().axes:
        ax.set_facecolor('black')
        ax.grid(False) # No grid
        # Set axis ticks to white
        ax.tick_params(axis='x', colors='k')
        ax.tick_params(axis='y', colors='k') 

    # plt.tight_layout()
        

    outputFile = OutputDir + 'SpatialDistribution' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    plt.close()

    # -------------------------------------------------------

    print('Plotting 3D spatial distribution with box and black background')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    w_box = np.where((Mvir > 0.0) & (StellarMass > 1.0e8) & (Type==0))[0]
    if(len(w_box) > dilute): w_box = sample(list(w_box), dilute)

    xx_box = Posx[w_box]
    yy_box = Posy[w_box]
    zz_box = Posz[w_box]
    logHotGas_box = np.log10(HotGas[w_box])
    logICS_box = np.log10(IntraClusterStars[w_box])

    # Plot Hot Gas (PuRd, faint, larger points)
    ax.scatter(xx_box, yy_box, zz_box, s=20, c=logHotGas_box, cmap='PuRd', alpha=0.07, edgecolors='none', label='Hot Gas')
    # Plot ICS (Blues, small, brighter points)
    ax.scatter(xx_box, yy_box, zz_box, s=0.2, c=logICS_box, cmap='Blues', alpha=0.8, edgecolors='none', label='ICS')

    # Draw the box edges
    points = np.array([[0,0,0], [BoxSize,0,0], [BoxSize,BoxSize,0], [0,BoxSize,0],
                       [0,0,BoxSize], [BoxSize,0,BoxSize], [BoxSize,BoxSize,BoxSize], [0,BoxSize,BoxSize]])

    edges = [[points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],
             [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],
             [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]]
    line_collection = Line3DCollection(edges, colors='k')
    ax.add_collection3d(line_collection)

    ax.set_xlabel('X (Mpc/h)')
    ax.set_ylabel('Y (Mpc/h)')
    ax.set_zlabel('Z (Mpc/h)')

    ax.set_xlim([0, BoxSize])
    ax.set_ylim([0, BoxSize])
    ax.set_zlim([0, BoxSize])

    # Set background color to black for the 3D plot
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.grid(False)
    ax.tick_params(axis='x', colors='k')
    ax.tick_params(axis='y', colors='k')
    ax.tick_params(axis='z', colors='k')

    ax.set_box_aspect([1,1,1])

    plt.tight_layout()

    outputFile_box = OutputDir + 'SpatialDistribution3D_Box' + OutputFormat
    plt.savefig(outputFile_box)
    print('Saved file to', outputFile_box, '\n')
    plt.close()
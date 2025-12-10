#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy import stats
from random import sample, seed

import warnings

# ========================== USER OPTIONS ==========================

DirName_FFB = './output/millennium/'
DirName_noFFB = './output/millennium_noffb/'
FileName = 'model_0.hdf5'
# List of snapshots to plot
# Snapshots = ['Snap_10', 'Snap_12', 'Snap_15', 'Snap_18', 'Snap_20', 'Snap_22']
# Snapshots = ['Snap_20', 'Snap_15', 'Snap_12', 'Snap_10', 'Snap_9', 'Snap_8']
Snapshots = ['Snap_22', 'Snap_20', 'Snap_18', 'Snap_15', 'Snap_12', 'Snap_11']
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

# Redshift list for snapshots
redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073, 
             9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 
             2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905, 
             0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 
             0.116, 0.089, 0.064, 0.041, 0.020, 0.000]
 
# ==================================================================

def read_hdf(file_name = None, snap_num = None, param = None, DirName=None):

    property = h5.File(DirName+file_name,'r')
    return np.array(property[snap_num][param])

def load_data(DirName, Snapshot, file_name=FileName):

    CentralMvir = read_hdf(snap_num = Snapshot, param = 'CentralMvir', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    Mvir = read_hdf(snap_num = Snapshot, param = 'Mvir', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    StellarMass = read_hdf(snap_num = Snapshot, param = 'StellarMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(snap_num = Snapshot, param = 'BulgeMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    BlackHoleMass = read_hdf(snap_num = Snapshot, param = 'BlackHoleMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    ColdGas = read_hdf(snap_num = Snapshot, param = 'ColdGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    MetalsColdGas = read_hdf(snap_num = Snapshot, param = 'MetalsColdGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    MetalsEjectedMass = read_hdf(snap_num = Snapshot, param = 'MetalsEjectedMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    HotGas = read_hdf(snap_num = Snapshot, param = 'HotGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    MetalsHotGas = read_hdf(snap_num = Snapshot, param = 'MetalsHotGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    EjectedMass = read_hdf(snap_num = Snapshot, param = 'EjectedMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    CGMgas = read_hdf(snap_num = Snapshot, param = 'CGMgas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    MetalsCGMgas = read_hdf(snap_num = Snapshot, param = 'MetalsCGMgas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    IntraClusterStars = read_hdf(snap_num = Snapshot, param = 'IntraClusterStars', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(snap_num = Snapshot, param = 'DiskRadius', DirName=DirName, file_name=file_name)
    BulgeRadius = read_hdf(snap_num = Snapshot, param = 'BulgeRadius', DirName=DirName, file_name=file_name)
    MergerBulgeRadius = read_hdf(snap_num = Snapshot, param = 'MergerBulgeRadius', DirName=DirName, file_name=file_name)
    InstabilityBulgeRadius = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeRadius', DirName=DirName, file_name=file_name)
    MergerBulgeMass = read_hdf(snap_num = Snapshot, param = 'MergerBulgeMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h

    H2gas = read_hdf(snap_num = Snapshot, param = 'H2gas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h
    Vvir = read_hdf(snap_num = Snapshot, param = 'Vvir', DirName=DirName, file_name=file_name)
    Vmax = read_hdf(snap_num = Snapshot, param = 'Vmax', DirName=DirName, file_name=file_name)
    Rvir = read_hdf(snap_num = Snapshot, param = 'Rvir', DirName=DirName, file_name=file_name)
    SfrDisk = read_hdf(snap_num = Snapshot, param = 'SfrDisk', DirName=DirName, file_name=file_name)
    SfrBulge = read_hdf(snap_num = Snapshot, param = 'SfrBulge', DirName=DirName, file_name=file_name)
    CentralGalaxyIndex = read_hdf(snap_num = Snapshot, param = 'CentralGalaxyIndex', DirName=DirName, file_name=file_name)
    Type = read_hdf(snap_num = Snapshot, param = 'Type', DirName=DirName, file_name=file_name)
    Posx = read_hdf(snap_num = Snapshot, param = 'Posx', DirName=DirName, file_name=file_name)
    Posy = read_hdf(snap_num = Snapshot, param = 'Posy', DirName=DirName, file_name=file_name)
    Posz = read_hdf(snap_num = Snapshot, param = 'Posz', DirName=DirName, file_name=file_name)

    OutflowRate = read_hdf(snap_num = Snapshot, param = 'OutflowRate', DirName=DirName, file_name=file_name)
    MassLoading = read_hdf(snap_num = Snapshot, param = 'MassLoading', DirName=DirName, file_name=file_name)
    Cooling = read_hdf(snap_num = Snapshot, param = 'Cooling', DirName=DirName, file_name=file_name)
    Tvir = 35.9 * (Vvir)**2  # in Kelvin
    Tmax = 2.5e5  # K, corresponds to Vvir ~52.7 km/s
    Regime = read_hdf(snap_num = Snapshot, param = 'Regime', DirName=DirName, file_name=file_name)
    ffb_regime = read_hdf(snap_num = Snapshot, param = 'FFBRegime', DirName=DirName, file_name=file_name)

    return {
        'CentralMvir': CentralMvir,
        'Mvir': Mvir,
        'StellarMass': StellarMass,
        'BulgeMass': BulgeMass,
        'BlackHoleMass': BlackHoleMass,
        'ColdGas': ColdGas,
        'MetalsColdGas': MetalsColdGas,
        'MetalsEjectedMass': MetalsEjectedMass,
        'HotGas': HotGas,
        'MetalsHotGas': MetalsHotGas,
        'EjectedMass': EjectedMass,
        'CGMgas': CGMgas,
        'MetalsCGMgas': MetalsCGMgas,
        'IntraClusterStars': IntraClusterStars,
        'DiskRadius': DiskRadius,
        'BulgeRadius': BulgeRadius,
        'MergerBulgeRadius': MergerBulgeRadius,
        'InstabilityBulgeRadius': InstabilityBulgeRadius,
        'MergerBulgeMass': MergerBulgeMass,
        'InstabilityBulgeMass': InstabilityBulgeMass,
        'H2gas': H2gas,
        'Vvir': Vvir,
        'Vmax': Vmax,
        'Rvir': Rvir,
        'SfrDisk': SfrDisk,
        'SfrBulge': SfrBulge,
        'CentralGalaxyIndex': CentralGalaxyIndex,
        'Type': Type,
        'Posx': Posx,
        'Posy': Posy,
        'Posz': Posz,
        'OutflowRate': OutflowRate,
        'MassLoading': MassLoading,
        'Cooling': Cooling,
        'Tvir': Tvir,
        'Tmax': Tmax,
        'Regime': Regime,
        'FFBRegime': ffb_regime
    }
def plot_sfr_distribution():
    """Plot SFR distribution for FFB and noFFB simulations at various snapshots."""

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)

    # Set up grid for 6 snapshots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for Snapshot, ax in zip(Snapshots, axes):
        print(f'Processing {Snapshot}')
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        Regime = data_FFB['Regime']
        ffb_regime = data_FFB['FFBRegime'] == 1



        # Match noFFB sample to FFB sample by Mvir
        mvir_FFB = data_FFB['Mvir'][ffb_regime]
        mvir_noFFB = data_noFFB['Mvir']
        stellar_mass_noFFB_full = data_noFFB['StellarMass']

        # Find closest Mvir matches (without replacement)
        idx_noFFB = np.arange(len(mvir_noFFB))
        used = set()
        matched_indices = []
        for m in mvir_FFB:
            diffs = np.abs(mvir_noFFB - m)
            # Mask out already used indices
            for i in used:
                diffs[i] = np.inf
            min_idx = np.argmin(diffs)
            matched_indices.append(min_idx)
            used.add(min_idx)

        stellar_mass_FFB = data_FFB['StellarMass'][ffb_regime]
        stellar_mass_noFFB = stellar_mass_noFFB_full[matched_indices]

        print(f"Snapshot {Snapshot}: tracking {len(stellar_mass_FFB)} galaxies with FFBs and {len(stellar_mass_noFFB)} galaxies without FFBs (Mvir-matched).")

        sfr_FFB = data_FFB['SfrDisk'][ffb_regime] + data_FFB['SfrBulge'][ffb_regime]
        sfr_noFFB = np.array(sample(list(data_noFFB['SfrDisk'] + data_noFFB['SfrBulge']), len(sfr_FFB)))
        log_sfr_FFB = np.log10(sfr_FFB + 1e-10)
        log_sfr_noFFB = np.log10(sfr_noFFB + 1e-10)

        binwidth = 0.2
        mi = np.floor(min(log_sfr_FFB)) - 2
        ma = np.floor(max(log_sfr_FFB)) + 2
        NB = int((ma - mi) / binwidth)
        (counts_FFB, binedges_FFB) = np.histogram(log_sfr_FFB, range=(mi, ma), bins=NB)
        xaxeshisto_FFB = binedges_FFB[:-1] + 0.5 * binwidth
        mi = np.floor(min(log_sfr_noFFB)) - 2
        ma = np.floor(max(log_sfr_noFFB)) + 2
        NB = int((ma - mi) / binwidth)
        (counts_noFFB, binedges_noFFB) = np.histogram(log_sfr_noFFB, range=(mi, ma), bins=NB)
        xaxeshisto_noFFB = binedges_noFFB[:-1] + 0.5 * binwidth 

        ax.plot(xaxeshisto_FFB, counts_FFB, drawstyle='steps-mid', label='With FFBs')
        ax.plot(xaxeshisto_noFFB, counts_noFFB, drawstyle='steps-mid', label='Without FFBs')
        # Remove individual axis labels; will set shared labels later
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        ax.set_title(f'z = {z:.3f}')
        ax.set_xlim(-4, 4)
        if axes.tolist().index(ax) == 0:
            ax.legend()


    # Set shared x-label and y-labels
    fig.supxlabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$', fontsize=20, y=0.02)
    fig.supylabel('Number of galaxies', fontsize=20, x=0.01)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(OutputDir + 'SFR_distribution_grid' + OutputFormat)
    plt.close()

def plot_sfr_distribution_scatter():
    """Plot SFR distribution for FFB and noFFB simulations at various snapshots."""

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)

    # Set up grid for 6 snapshots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for Snapshot, ax in zip(Snapshots, axes):
        print(f'Processing {Snapshot}')
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        Regime = data_FFB['Regime']
        ffb_regime = data_FFB['FFBRegime'] == 1



        # Match noFFB sample to FFB sample by Mvir
        mvir_FFB = data_FFB['Mvir'][ffb_regime]
        mvir_noFFB = data_noFFB['Mvir']
        stellar_mass_noFFB_full = data_noFFB['StellarMass']

        # Find closest Mvir matches (without replacement)
        idx_noFFB = np.arange(len(mvir_noFFB))
        used = set()
        matched_indices = []
        for m in mvir_FFB:
            diffs = np.abs(mvir_noFFB - m)
            # Mask out already used indices
            for i in used:
                diffs[i] = np.inf
            min_idx = np.argmin(diffs)
            matched_indices.append(min_idx)
            used.add(min_idx)

        stellar_mass_FFB = np.log10(data_FFB['StellarMass'][ffb_regime])
        stellar_mass_noFFB = np.log10(stellar_mass_noFFB_full[matched_indices])

        print(f"Snapshot {Snapshot}: tracking {len(stellar_mass_FFB)} galaxies with FFBs and {len(stellar_mass_noFFB)} galaxies without FFBs (Mvir-matched).")

        sfr_FFB = data_FFB['SfrDisk'][ffb_regime] + data_FFB['SfrBulge'][ffb_regime]
        sfr_noFFB = np.array(sample(list(data_noFFB['SfrDisk'] + data_noFFB['SfrBulge']), len(sfr_FFB)))
        log_sfr_FFB = np.log10(sfr_FFB + 1e-10)
        log_sfr_noFFB = np.log10(sfr_noFFB + 1e-10)
 
        ax.scatter(stellar_mass_FFB, log_sfr_FFB, label='With FFBs', alpha=0.5, color='C0')
        ax.scatter(stellar_mass_noFFB, log_sfr_noFFB, label='Without FFBs', alpha=0.5, color='C1')

        # Remove individual axis labels; will set shared labels later
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        ax.set_title(f'z = {z:.3f}')
        # ax.set_xlim(-4, 4)
        if axes.tolist().index(ax) == 0:
            ax.legend()


    # Set shared x-label and y-labels
    fig.supxlabel(r'$\log_{10}(\mathrm{m}_{\star}\ [M_{\odot}])$', fontsize=20, y=0.02)
    fig.supylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$', fontsize=20, x=0.01)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(OutputDir + 'SFR_distribution_grid_scatter' + OutputFormat)
    plt.close()

def plot_metallicity_distribution():
    """Plot metallicity distribution for FFB and noFFB simulations at various snapshots."""

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)

    # Set up grid for 6 snapshots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for Snapshot, ax in zip(Snapshots, axes):
        print(f'Processing {Snapshot}')
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        Regime = data_FFB['Regime']
        ffb_regime = data_FFB['FFBRegime'] == 1

        type_ffb = data_FFB['Type']
        type_noffb = data_noFFB['Type']



        # Match noFFB sample to FFB sample by Mvir
        mvir_FFB = data_FFB['Mvir'][ffb_regime]
        mvir_noFFB = data_noFFB['Mvir']
        coldgas_mass_noFFB_full = data_noFFB['ColdGas']
        metals_coldgas_noFFB_full = data_noFFB['MetalsColdGas']
        stellar_mass_noFFB_full = data_noFFB['StellarMass']

        # Find closest Mvir matches (without replacement)
        idx_noFFB = np.arange(len(mvir_noFFB))
        used = set()
        matched_indices = []
        for m in mvir_FFB:
            diffs = np.abs(mvir_noFFB - m)
            # Mask out already used indices
            for i in used:
                diffs[i] = np.inf
            min_idx = np.argmin(diffs)
            matched_indices.append(min_idx)
            used.add(min_idx)

        stellar_mass_FFB = data_FFB['StellarMass'][ffb_regime]
        coldgas_mass_FFB = data_FFB['ColdGas'][ffb_regime]
        metals_coldgas_FFB = data_FFB['MetalsColdGas'][ffb_regime]
        coldgas_mass_noFFB = coldgas_mass_noFFB_full[matched_indices]
        metals_coldgas_noFFB = metals_coldgas_noFFB_full[matched_indices]
        stellar_mass_noFFB = stellar_mass_noFFB_full[matched_indices]

        print(f"Snapshot {Snapshot}: tracking {len(stellar_mass_FFB)} galaxies with FFBs and {len(stellar_mass_noFFB)} galaxies without FFBs (Mvir-matched).")

        w = np.where((coldgas_mass_noFFB / (stellar_mass_noFFB + coldgas_mass_noFFB) > 0.1) & (stellar_mass_noFFB > 1.0e8))[0]
        w2 = np.where((coldgas_mass_FFB / (stellar_mass_FFB + coldgas_mass_FFB) > 0.1) & (stellar_mass_FFB > 1.0e8))[0]
        
        mass_noffb = np.log10(stellar_mass_noFFB[w])
        mass_ffb = np.log10(stellar_mass_FFB[w2])
        Z_FFB = np.log10((metals_coldgas_FFB[w2] / coldgas_mass_FFB[w2]) / 0.02) + 9.0
        Z_noFFB = np.log10((metals_coldgas_noFFB[w] / coldgas_mass_noFFB[w]) / 0.02) + 9.0

        # print(f"Sample metallicities (FFB): {Z_FFB[:5]}")
        # print(f"Sample metallicities (noFFB): {Z_noFFB[:5]}")

        binwidth = 0.2
        mi = np.floor(min(Z_FFB)) - 2
        ma = np.floor(max(Z_FFB)) + 2
        NB = int((ma - mi) / binwidth)
        (counts_FFB, binedges_FFB) = np.histogram(Z_FFB, range=(mi, ma), bins=NB)
        xaxeshisto_FFB = binedges_FFB[:-1] + 0.5 * binwidth
        mi = np.floor(min(Z_noFFB)) - 2
        ma = np.floor(max(Z_noFFB)) + 2
        NB = int((ma - mi) / binwidth)
        (counts_noFFB, binedges_noFFB) = np.histogram(Z_noFFB, range=(mi, ma), bins=NB)
        xaxeshisto_noFFB = binedges_noFFB[:-1] + 0.5 * binwidth

        ax.plot(xaxeshisto_FFB, counts_FFB, drawstyle='steps-mid', label='With FFBs')
        ax.plot(xaxeshisto_noFFB, counts_noFFB, drawstyle='steps-mid', label='Without FFBs')


        # Remove individual axis labels; will set shared labels later
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        ax.set_title(f'z = {z:.3f}')
        # ax.set_xlim(-4, 4)
        if axes.tolist().index(ax) == 0:
            ax.legend()


    # Set shared x-label and y-labels
    # fig.supxlabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$', fontsize=20, y=0.02)
    # fig.supylabel('Number of galaxies', fontsize=20, x=0.01)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(OutputDir + 'metallicity_distribution_grid' + OutputFormat)
    plt.close()


# ==================================================================

if __name__ == "__main__":

    plot_sfr_distribution()
    plot_sfr_distribution_scatter()
    plot_metallicity_distribution()
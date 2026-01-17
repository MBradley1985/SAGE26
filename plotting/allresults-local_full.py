#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy import stats
from random import sample, seed

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
    MergerBulgeRadius = read_hdf(snap_num = Snapshot, param = 'MergerBulgeRadius')
    InstabilityBulgeRadius = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeRadius')
    MergerBulgeMass = read_hdf(snap_num = Snapshot, param = 'MergerBulgeMass') * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeMass') * 1.0e10 / Hubble_h

    print("Bulge Scale Radius sample:")
    print(BulgeRadius)

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


    w = np.where(StellarMass > 1.0e10)[0]
    print('Number of galaxies read:', len(StellarMass))
    print('Galaxies more massive than 10^10 h-1 Msun:', len(w), '\n')

    Cooling = read_hdf(snap_num = Snapshot, param = 'Cooling')

    Tvir = 35.9 * (Vvir)**2  # in Kelvin
    Tmax = 2.5e5  # K, corresponds to Vvir ~52.7 km/s

    Regime = read_hdf(snap_num = Snapshot, param = 'Regime')

# --------------------------------------------------------

    print('Plotting the stellar mass function, divided by sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width
    DirName2 = './output/millennium_vanilla/'

    # Load GAMA morphological SMF data
    # Columns: log_M, E_HE, E_HE_err, cBD, cBD_err, dBD, dBD_err, D, D_err
    gama = np.genfromtxt('./data/gama_smf_morph.ecsv', comments='#', skip_header=1)
    gama_mass = gama[:, 0]
    gama_E_HE = gama[:, 1]
    gama_E_HE_err = gama[:, 2]
    gama_D = gama[:, 7]
    gama_D_err = gama[:, 8]

    # Load Baldry et al. blue/red SMF data
    # Columns: SF_mass, SF_phi, Q_mass, Q_phi (all in log)
    baldry = np.genfromtxt('./data/baldry_blue_red.csv', delimiter=',', skip_header=2)
    baldry_sf_mass = baldry[:, 0]
    baldry_sf_phi = baldry[:, 1]
    baldry_q_mass = baldry[:, 2]
    baldry_q_phi = baldry[:, 3]

    smass_vanilla = read_hdf(filename = DirName2+FileName, snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
    sfrdisk_vanilla = read_hdf(filename = DirName2+FileName, snap_num = Snapshot, param = 'SfrDisk')
    sfrbulge_vanilla = read_hdf(filename = DirName2+FileName, snap_num = Snapshot, param = 'SfrBulge')

    # calculate all
    w = np.where(StellarMass > 0.0)[0]
    mass = np.log10(StellarMass[w])
    sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )

    w2 = np.where(smass_vanilla > 0.0)[0]
    mass_vanilla = np.log10(smass_vanilla[w2])
    sSFR_vanilla = np.log10( (sfrdisk_vanilla[w2] + sfrbulge_vanilla[w2]) / smass_vanilla[w2] )

    # Bin parameters for original model
    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # additionally calculate red for original model
    w = np.where(sSFR < sSFRcut)[0]
    massRED = mass[w]
    (countsRED, binedges) = np.histogram(massRED, range=(mi, ma), bins=NB)

    # additionally calculate blue for original model
    w = np.where(sSFR > sSFRcut)[0]
    massBLU = mass[w]
    (countsBLU, binedges) = np.histogram(massBLU, range=(mi, ma), bins=NB)

    # Bin parameters for vanilla model
    mi_v = np.floor(min(mass_vanilla)) - 2
    ma_v = np.floor(max(mass_vanilla)) + 2
    NB_v = int((ma_v - mi_v) / binwidth)
    (counts_vanilla, binedges) = np.histogram(mass_vanilla, range=(mi_v, ma_v), bins=NB_v)
    xaxeshisto_vanilla = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # additionally calculate red for vanilla
    w2 = np.where(sSFR_vanilla < sSFRcut)[0]
    massRED_vanilla = mass_vanilla[w2]
    (countsRED_vanilla, binedges) = np.histogram(massRED_vanilla, range=(mi_v, ma_v), bins=NB_v)

    # additionally calculate blue for vanilla
    w2 = np.where(sSFR_vanilla > sSFRcut)[0]
    massBLU_vanilla = mass_vanilla[w2]
    (countsBLU_vanilla, binedges) = np.histogram(massBLU_vanilla, range=(mi_v, ma_v), bins=NB_v)


    # Overplot the model histograms (in log10 space)
    # plt.plot(xaxeshisto, np.log10(counts / volume / binwidth), 'k-', label='SAGE26')
    plt.plot(xaxeshisto, np.log10(counts / volume / binwidth), color='black', lw=4, label='SAGE26 Total')
    plt.plot(xaxeshisto, np.log10(countsRED / volume / binwidth), color='firebrick', lw=4, label='SAGE26 Quiescent')
    plt.plot(xaxeshisto, np.log10(countsBLU / volume / binwidth), color='dodgerblue', lw=4, label='SAGE26 Star Forming')

    plt.plot(xaxeshisto_vanilla, np.log10(countsRED_vanilla / volume / binwidth), color='firebrick', lw=2, ls='--', label='C16 Quiescent')
    plt.plot(xaxeshisto_vanilla, np.log10(countsBLU_vanilla / volume / binwidth), color='dodgerblue', lw=2, ls='--', label='C16 Star Forming')

    # Create shaded regions from observations (GAMA + Baldry combined)
    from scipy import interpolate

    # Common mass grid for interpolation
    mass_grid = np.linspace(8, 12, 100)

    # Star-forming: combine GAMA D and Baldry SF
    valid_D = ~np.isnan(gama_D)
    gama_sf_interp = interpolate.interp1d(gama_mass[valid_D], gama_D[valid_D],
                                           bounds_error=False, fill_value=np.nan)
    baldry_sf_interp = interpolate.interp1d(baldry_sf_mass, baldry_sf_phi,
                                             bounds_error=False, fill_value=np.nan)
    sf_gama = gama_sf_interp(mass_grid)
    sf_baldry = baldry_sf_interp(mass_grid)
    sf_lower = np.nanmin([sf_gama, sf_baldry], axis=0)
    sf_upper = np.nanmax([sf_gama, sf_baldry], axis=0)

    # Quiescent: combine GAMA E+HE and Baldry Q
    valid_E = ~np.isnan(gama_E_HE)
    gama_q_interp = interpolate.interp1d(gama_mass[valid_E], gama_E_HE[valid_E],
                                          bounds_error=False, fill_value=np.nan)
    baldry_q_interp = interpolate.interp1d(baldry_q_mass, baldry_q_phi,
                                            bounds_error=False, fill_value=np.nan)
    q_gama = gama_q_interp(mass_grid)
    q_baldry = baldry_q_interp(mass_grid)
    q_lower = np.nanmin([q_gama, q_baldry], axis=0)
    q_upper = np.nanmax([q_gama, q_baldry], axis=0)

    # Plot shaded regions
    plt.fill_between(mass_grid, sf_lower, sf_upper, color='dodgerblue', alpha=0.3, edgecolor='none', label='Observations SF')
    plt.fill_between(mass_grid, q_lower, q_upper, color='firebrick', alpha=0.3, edgecolor='none', label='Observations Q')

    plt.axis([8, 12, -6, -1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.ylabel(r'$\log_{10}\ \phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    outputFile = OutputDir + '1.StellarMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the baryonic mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width
  
    # calculate BMF
    w = np.where(StellarMass + ColdGas > 0.0)[0]
    mass = np.log10( (StellarMass[w] + ColdGas[w]) )

    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    centrals = np.where(Type[w] == 0)[0]
    satellites = np.where(Type[w] == 1)[0]

    centrals_mass = mass[centrals]
    satellites_mass = mass[satellites]

    mi = np.floor(min(centrals_mass)) - 2
    ma = np.floor(max(centrals_mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts_centrals, binedges_centrals) = np.histogram(centrals_mass, range=(mi, ma), bins=NB)
    xaxeshisto_centrals = binedges_centrals[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    mi = np.floor(min(satellites_mass)) - 2
    ma = np.floor(max(satellites_mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts_satellites, binedges_satellites) = np.histogram(satellites_mass, range=(mi, ma), bins=NB)
    xaxeshisto_satellites = binedges_satellites[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # Bell et al. 2003 BMF (h=1.0 converted to h=0.73)
    M = np.arange(7.0, 13.0, 0.01)
    Mstar = np.log10(5.3*1.0e10 /Hubble_h/Hubble_h)
    alpha = -1.21
    phistar = 0.0108 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    
    if(whichimf == 0):
        # converted diet Salpeter IMF to Salpeter IMF
        plt.plot(np.log10(10.0**M /0.7), yval, 'b-', lw=2.0, label='Bell et al. 2003')  # Plot the SMF
    elif(whichimf == 1):
        # converted diet Salpeter IMF to Salpeter IMF, then to Chabrier IMF
        plt.plot(np.log10(10.0**M /0.7 /1.8), yval, 'g--', lw=1.5, label='Bell et al. 2003')  # Plot the SMF

    # Overplot the model histograms
    plt.plot(xaxeshisto, counts / volume / binwidth, 'k-', label='Model')
    plt.plot(xaxeshisto_centrals, counts_centrals / volume / binwidth, 'b:', lw=2, label='Model - Centrals')
    plt.plot(xaxeshisto_satellites, counts_satellites / volume / binwidth, 'g--', lw=1.5, label='Model - Satellites')

    plt.yscale('log')
    plt.axis([8.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
    
    outputFile = OutputDir + '2.BaryonicMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the cold gas mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

    # calculate all
    w = np.where((ColdGas > 0.0) & (Type==0))[0]
    mass = np.log10(ColdGas[w])
    H2mass = np.log10(H2gas[w])
    H1mass = np.log10(H1gas[w])  # Now read directly from model output
    sSFR = (SfrDisk[w] + SfrBulge[w]) / StellarMass[w]

    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)

    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    (counts_h2, binedges_h2) = np.histogram(H2mass, range=(mi, ma), bins=NB)
    xaxeshisto_h2 = binedges_h2[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    (counts_h1, binedges_h1) = np.histogram(H1mass, range=(mi, ma), bins=NB)
    xaxeshisto_h1 = binedges_h1[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # additionally calculate red
    w = np.where(sSFR < sSFRcut)[0]
    massRED = mass[w]
    (countsRED, binedges) = np.histogram(massRED, range=(mi, ma), bins=NB)

    # additionally calculate blue
    w = np.where(sSFR > sSFRcut)[0]
    massBLU = mass[w]
    (countsBLU, binedges) = np.histogram(massBLU, range=(mi, ma), bins=NB)

    # Baldry+ 2008 modified data used for the MCMC fitting
    Zwaan = np.array([[6.933,   -0.333],
        [7.057,   -0.490],
        [7.209,   -0.698],
        [7.365,   -0.667],
        [7.528,   -0.823],
        [7.647,   -0.958],
        [7.809,   -0.917],
        [7.971,   -0.948],
        [8.112,   -0.927],
        [8.263,   -0.917],
        [8.404,   -1.062],
        [8.566,   -1.177],
        [8.707,   -1.177],
        [8.853,   -1.312],
        [9.010,   -1.344],
        [9.161,   -1.448],
        [9.302,   -1.604],
        [9.448,   -1.792],
        [9.599,   -2.021],
        [9.740,   -2.406],
        [9.897,   -2.615],
        [10.053,  -3.031],
        [10.178,  -3.677],
        [10.335,  -4.448],
        [10.492,  -5.083]        ], dtype=np.float32)
    
    ObrRaw = np.array([
        [7.300,   -1.104],
        [7.576,   -1.302],
        [7.847,   -1.250],
        [8.133,   -1.240],
        [8.409,   -1.344],
        [8.691,   -1.479],
        [8.956,   -1.792],
        [9.231,   -2.271],
        [9.507,   -3.198],
        [9.788,   -5.062 ]        ], dtype=np.float32)
    ObrCold = np.array([
        [8.009,   -1.042],
        [8.215,   -1.156],
        [8.409,   -0.990],
        [8.604,   -1.156],
        [8.799,   -1.208],
        [9.020,   -1.333],
        [9.194,   -1.385],
        [9.404,   -1.552],
        [9.599,   -1.677],
        [9.788,   -1.812],
        [9.999,   -2.312],
        [10.172,  -2.656],
        [10.362,  -3.500],
        [10.551,  -3.635],
        [10.740,  -5.010]        ], dtype=np.float32)
    
    ObrCold_xval = np.log10(10**(ObrCold[:, 0])  /Hubble_h/Hubble_h)
    ObrCold_yval = (10**(ObrCold[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    Zwaan_xval = np.log10(10**(Zwaan[:, 0]) /Hubble_h/Hubble_h)
    Zwaan_yval = (10**(Zwaan[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    ObrRaw_xval = np.log10(10**(ObrRaw[:, 0])  /Hubble_h/Hubble_h)
    ObrRaw_yval = (10**(ObrRaw[:, 1]) * Hubble_h*Hubble_h*Hubble_h)

    plt.plot(ObrCold_xval, ObrCold_yval, color='black', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (Cold Gas)')
    plt.plot(Zwaan_xval, Zwaan_yval, color='cyan', lw = 7, alpha=0.25, label='Zwaan et al. 2005 (HI)')
    plt.plot(ObrRaw_xval, ObrRaw_yval, color='magenta', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (H2)')

    plt.plot(xaxeshisto_h2, counts_h2 / volume / binwidth, 'magenta', linestyle='-', label='Model - H2 Gas')
    plt.plot(xaxeshisto_h1, counts_h1 / volume / binwidth, 'cyan', linestyle='-', label='Model - HI Gas')
    
    # Overplot the model histograms
    plt.plot(xaxeshisto, counts / volume / binwidth, 'k-', label='Model - Cold Gas')

    plt.yscale('log')
    plt.axis([8.0, 11.5, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{X}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    outputFile = OutputDir + '3.GasMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the baryonic TF relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (StellarMass + ColdGas > 0.0) & 
      (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
    if(len(w) > dilute): w = sample(list(range(len(w))), dilute)

    mass = np.log10( (StellarMass[w] + ColdGas[w]) )
    vel = np.log10(Vmax[w])
                
    plt.scatter(vel, mass, marker='x', s=50, c='k', alpha=0.3, label='Model Sb/c galaxies')
            
    # overplot Stark, McGaugh & Swatters 2009 (assumes h=0.75? ... what IMF?)
    w = np.arange(0.5, 10.0, 0.5)
    TF = 3.94*w + 1.79
    TF_upper = TF + 0.26
    TF_lower = TF - 0.26

    # plt.plot(w, TF, 'b-', alpha=0.5, label='Stark, McGaugh & Swatters 2009')
    plt.fill_between(w, TF_lower, TF_upper, color='blue', alpha=0.2)

        
    plt.ylabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log_{10}V_{max}\ (km/s)$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([1.4, 2.9, 7.5, 12.0])
        
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
        
    outputFile = OutputDir + '4.BaryonicTullyFisher' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the specific sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where(StellarMass > 0.01)[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    mass = np.log10(StellarMass[w])
    sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )
    plt.scatter(mass, sSFR, marker='o', s=1, c='k', alpha=0.5, label='Model galaxies')

    # overplot dividing line between SF and passive
    w = np.arange(7.0, 13.0, 1.0)
    plt.plot(w, w/w*sSFRcut, 'b:', lw=2.0)

    plt.ylabel(r'$\log_{10}\ s\mathrm{SFR}\ (\mathrm{yr^{-1}})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([8.0, 12.0, -16.0, -8.0])

    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    outputFile = OutputDir + '5.SpecificStarFormationRate' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the gas fractions')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (StellarMass + ColdGas > 0.0) & 
      (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    
    mass = np.log10(StellarMass[w])
    fraction = ColdGas[w] / (StellarMass[w] + ColdGas[w])

    plt.scatter(mass, fraction, marker='o', s=1, c='k', alpha=0.5, label='Model Sb/c galaxies')
        
    plt.ylabel(r'$\mathrm{Cold\ Mass\ /\ (Cold+Stellar\ Mass)}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([8.0, 12.0, 0.0, 1.0])
        
    leg = plt.legend(loc='upper right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
        
    outputFile = OutputDir + '6.GasFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the metallicities')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (ColdGas / (StellarMass + ColdGas) > 0.1) & (StellarMass > 1.0e8))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    
    mass = np.log10(StellarMass[w])
    Z = np.log10((MetalsColdGas[w] / ColdGas[w]) / 0.02) + 9.0
    
    plt.scatter(mass, Z, marker='o', s=1, c='k', alpha=0.5, label='Model galaxies')
        
    # overplot Tremonti et al. 2003 (h=0.7)
    w = np.arange(7.0, 11.5, 0.1)
    Zobs = -1.492 + 1.847*w - 0.08026*w*w
    if(whichimf == 0):
        # Conversion from Kroupa IMF to Slapeter IMF
        # plt.plot(np.log10((10**w *1.5)), Zobs, 'b-', lw=2.0, label='Tremonti et al. 2003')
        plt.fill_between(np.log10((10**w *1.5)), Zobs+0.1, Zobs-0.1, color='blue', alpha=0.2)
    elif(whichimf == 1):
        # Conversion from Kroupa IMF to Slapeter IMF to Chabrier IMF
        # plt.plot(np.log10((10**w *1.5 /1.8)), Zobs, 'b-', lw=2.0, label='Tremonti et al. 2003')
        plt.fill_between(np.log10((10**w *1.5 /1.8)), Zobs+0.1, Zobs-0.1, color='blue', alpha=0.2)
        
    plt.ylabel(r'$12\ +\ \log_{10}(\mathrm{O/H})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        
    plt.axis([8.0, 12.0, 8.0, 9.5])
        
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
        
    outputFile = OutputDir + '7.Metallicity' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the black hole-bulge relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((BulgeMass > 1.0e8) & (BlackHoleMass > 01.0e6))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    bh = np.log10(BlackHoleMass[w])
    bulge = np.log10(BulgeMass[w])
                
    plt.scatter(bulge, bh, marker='o', s=1, c='k', alpha=0.5, label='Model galaxies')
            
    # overplot Haring & Rix 2004
    w = 10. ** np.arange(20)
    BHdata = 10. ** (8.2 + 1.12 * np.log10(w / 1.0e11))
    plt.plot(np.log10(w), np.log10(BHdata), 'b-', label="Haring \& Rix 2004")

    # Observational points
    M_BH_obs = (0.7/Hubble_h)**2*1e8*np.array([39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39, 0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12, 1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4, 0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5, 0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06, 0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10, 3.7, 0.55, 13, 0.11])
    M_BH_hi = (0.7/Hubble_h)**2*1e8*np.array([4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26, 0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005, 0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3, 0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3, 0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2, 0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004, 0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = (0.7/Hubble_h)**2*1e8*np.array([5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09, 0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005, 0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3, 0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5, 0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1, 0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004, 0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = (0.7/Hubble_h)**2*1e10*np.array([69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9, 4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88, 1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6, 3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9, 1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29, 11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = (0.7/Hubble_h)**2*1e10*np.array([59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7, 6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28, 2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8, 2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2, 6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10, 17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = (0.7/Hubble_h)**2*1e10*np.array([32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1, 2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52, 1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5, 1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6, 3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9, 1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,1,0])
    yerr2, yerr1 = np.log10((M_BH_obs+M_BH_hi)/M_BH_obs), -np.log10((M_BH_obs-M_BH_lo)/M_BH_obs)
    xerr2, xerr1 = np.log10((M_sph_obs+M_sph_hi)/M_sph_obs), -np.log10((M_sph_obs-M_sph_lo)/M_sph_obs)
    plt.errorbar(np.log10(M_sph_obs[core==0]), np.log10(M_BH_obs[core==0]), yerr=[yerr1[core==0],yerr2[core==0]], xerr=[xerr1[core==0],xerr2[core==0]], color='purple', alpha=0.3, label=r'S13 core', ls='none', lw=2, ms=0)
    plt.errorbar(np.log10(M_sph_obs[core==1]), np.log10(M_BH_obs[core==1]), yerr=[yerr1[core==1],yerr2[core==1]], xerr=[xerr1[core==1],xerr2[core==1]], color='c', alpha=0.3, label=r'S13 Sersic', ls='none', lw=2, ms=0)
    
    plt.ylabel(r'$\log\ M_{\mathrm{BH}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log\ M_{\mathrm{bulge}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([8.0, 12.0, 6.0, 10.0])
        
    leg = plt.legend(loc='upper left')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
        
    outputFile = OutputDir + '8.BlackHoleBulgeRelationship' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the quiescent fraction vs stellar mass')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure
    
    groupscale = 12.5
    
    w = np.where(StellarMass > 0.0)[0]
    stars = np.log10(StellarMass[w])
    halo = np.log10(CentralMvir[w])
    galtype = Type[w]
    sSFR = (SfrDisk[w] + SfrBulge[w]) / StellarMass[w]

    MinRange = 9.5
    MaxRange = 12.5
    Interval = 0.1
    Nbins = int((MaxRange-MinRange)/Interval)
    Range = np.arange(MinRange, MaxRange, Interval)
    
    Mass = []
    Fraction = []
    CentralFraction = []
    SatelliteFraction = []
    SatelliteFractionLo = []
    SatelliteFractionHi = []

    for i in range(Nbins-1):
        
        w = np.where((stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            Fraction.append(1.0*len(wQ) / len(w))
        else:
            Fraction.append(0.0)
        
        w = np.where((galtype == 0) & (stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((galtype == 0) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            CentralFraction.append(1.0*len(wQ) / len(w))
        else:
            CentralFraction.append(0.0)
        
        w = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            SatelliteFraction.append(1.0*len(wQ) / len(w))
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut) & (halo < groupscale))[0]
            SatelliteFractionLo.append(1.0*len(wQ) / len(w))
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut) & (halo > groupscale))[0]
            SatelliteFractionHi.append(1.0*len(wQ) / len(w))                
        else:
            SatelliteFraction.append(0.0)
            SatelliteFractionLo.append(0.0)
            SatelliteFractionHi.append(0.0)
            
        Mass.append((Range[i] + Range[i+1]) / 2.0)                
    
    Mass = np.array(Mass)
    Fraction = np.array(Fraction)
    CentralFraction = np.array(CentralFraction)
    SatelliteFraction = np.array(SatelliteFraction)
    SatelliteFractionLo = np.array(SatelliteFractionLo)
    SatelliteFractionHi = np.array(SatelliteFractionHi)
    
    w = np.where(Fraction > 0)[0]
    plt.plot(Mass[w], Fraction[w], label='All')

    w = np.where(CentralFraction > 0)[0]
    plt.plot(Mass[w], CentralFraction[w], color='Blue', label='Centrals')

    w = np.where(SatelliteFraction > 0)[0]
    plt.plot(Mass[w], SatelliteFraction[w], color='Red', label='Satellites')

    w = np.where(SatelliteFractionLo > 0)[0]
    plt.plot(Mass[w], SatelliteFractionLo[w], 'r--', label='Satellites-Lo')

    w = np.where(SatelliteFractionHi > 0)[0]
    plt.plot(Mass[w], SatelliteFractionHi[w], 'r-.', label='Satellites-Hi')
    
    plt.xlabel(r'$\log_{10} M_{\mathrm{stellar}}\ (M_{\odot})$')  # Set the x-axis label
    plt.ylabel(r'$\mathrm{Quescient\ Fraction}$')  # Set the y-axis label
    
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    
    plt.axis([9.5, 12.0, 0.0, 1.05])
    
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
    
    outputFile = OutputDir + '9.QuiescentFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the mass fraction of galaxies')

    w = np.where(StellarMass > 0.0)[0]
    fBulge = BulgeMass[w] / StellarMass[w]
    fDisk = 1.0 - fBulge
    mass = np.log10(StellarMass[w])
    sSFR = np.log10((SfrDisk[w] + SfrBulge[w]) / StellarMass[w])
    
    binwidth = 0.2
    shift = binwidth/2.0
    mass_range = np.arange(8.5-shift, 12.0+shift, binwidth)
    bins = len(mass_range)
    
    fBulge_ave = np.zeros(bins)
    fBulge_var = np.zeros(bins)
    fDisk_ave = np.zeros(bins)
    fDisk_var = np.zeros(bins)
    
    for i in range(bins-1):
        w = np.where( (mass >= mass_range[i]) & (mass < mass_range[i+1]))[0]
        if(len(w) > 0):
            fBulge_ave[i] = np.mean(fBulge[w])
            fBulge_var[i] = np.var(fBulge[w])
            fDisk_ave[i] = np.mean(fDisk[w])
            fDisk_var[i] = np.var(fDisk[w])
    
    w = np.where(fBulge_ave > 0.0)[0]
    plt.plot(mass_range[w]+shift, fBulge_ave[w], 'r-', label='bulge')
    plt.fill_between(mass_range[w]+shift, 
        fBulge_ave[w]+fBulge_var[w], 
        fBulge_ave[w]-fBulge_var[w], 
        facecolor='red', alpha=0.25)
    
    w = np.where(fDisk_ave > 0.0)[0]
    plt.plot(mass_range[w]+shift, fDisk_ave[w], 'k-', label='disk stars')
    plt.fill_between(mass_range[w]+shift, 
        fDisk_ave[w]+fDisk_var[w], 
        fDisk_ave[w]-fDisk_var[w], 
        facecolor='black', alpha=0.25)
    
    plt.axis([mass_range[0], mass_range[bins-1], 0.0, 1.05])

    plt.ylabel(r'$\mathrm{Stellar\ Mass\ Fraction}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='upper right', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')
    
    outputFile = OutputDir + '10.BulgeMassFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the average baryon fraction vs halo mass (can take time)')

    # Find halos at log Mvir = 13.5-14.0
    mask = (np.log10(Mvir) > 13.5) & (np.log10(Mvir) < 14.0)

    total_baryons = (StellarMass[mask] + ColdGas[mask] + HotGas[mask] + CGMgas[mask] + IntraClusterStars[mask] + BlackHoleMass[mask] + EjectedMass[mask]) / (0.17 * Mvir[mask])
    print(f"Baryon closure at high mass: {np.mean(total_baryons):.3f}")
    print(f"Should be ~1.0. If < 0.95, baryons are leaking somewhere.")

    # Check component fractions
    print(f"Hot gas fraction: {np.mean(HotGas[mask] / (0.17 * Mvir[mask])):.3f}")
    print(f"Stellar fraction: {np.mean(StellarMass[mask] / (0.17 * Mvir[mask])):.3f}")
    print(f"CGM fraction: {np.mean(CGMgas[mask] / (0.17 * Mvir[mask])):.3f}  # Should be ~0")

    plt.figure()
    ax = plt.subplot(111)

    HaloMass = np.log10(Mvir)
    Baryons = StellarMass + ColdGas + HotGas + CGMgas + IntraClusterStars + BlackHoleMass + EjectedMass

    MinHalo, MaxHalo, Interval = 11.0, 16.0, 0.1
    HaloBins = np.arange(MinHalo, MaxHalo + Interval, Interval)
    Nbins = len(HaloBins) - 1

    MeanCentralHaloMass = []
    MeanBaryonFraction = []
    MeanBaryonFractionU = []
    MeanBaryonFractionL = []
    MeanStars = []
    MeanStarsU = []
    MeanStarsL = []
    MeanCold = []
    MeanColdU = []
    MeanColdL = []
    MeanHot = []
    MeanHotU = []
    MeanHotL = []
    MeanCGM = []
    MeanCGMU = []
    MeanCGML = []
    MeanICS = []
    MeanICSU = []
    MeanICSL = []
    MeanBH = []
    MeanBHU = []
    MeanBHL = []
    MeanEjected = []
    MeanEjectedU = []
    MeanEjectedL = []

    bin_indices = np.digitize(HaloMass, HaloBins) - 1

    # Pre-compute unique CentralGalaxyIndex for faster lookup
    halo_to_galaxies = defaultdict(list)
    for i, central_idx in enumerate(CentralGalaxyIndex):
        halo_to_galaxies[central_idx].append(i)

    for i in range(Nbins - 1):
        w1 = np.where((Type == 0) & (bin_indices == i))[0]
        HalosFound = len(w1)
        
        if HalosFound > 2:
            # Pre-allocate arrays for better performance
            BaryonFractions = np.zeros(HalosFound)
            StarsFractions = np.zeros(HalosFound)
            ColdFractions = np.zeros(HalosFound)
            HotFractions = np.zeros(HalosFound)
            CGMFractions = np.zeros(HalosFound)
            ICSFractions = np.zeros(HalosFound)
            BHFractions = np.zeros(HalosFound)
            EjectedFractions = np.zeros(HalosFound)
            
            # Vectorized calculation for each halo
            for idx, halo_idx in enumerate(w1):
                halo_galaxies = np.array(halo_to_galaxies[CentralGalaxyIndex[halo_idx]])
                halo_mvir = Mvir[halo_idx]
                
                # Use advanced indexing for faster summing
                BaryonFractions[idx] = np.sum(Baryons[halo_galaxies]) / halo_mvir
                StarsFractions[idx] = np.sum(StellarMass[halo_galaxies]) / halo_mvir
                ColdFractions[idx] = np.sum(ColdGas[halo_galaxies]) / halo_mvir
                HotFractions[idx] = np.sum(HotGas[halo_galaxies]) / halo_mvir
                CGMFractions[idx] = np.sum(CGMgas[halo_galaxies]) / halo_mvir
                ICSFractions[idx] = np.sum(IntraClusterStars[halo_galaxies]) / halo_mvir
                BHFractions[idx] = np.sum(BlackHoleMass[halo_galaxies]) / halo_mvir
                EjectedFractions[idx] = np.sum(EjectedMass[halo_galaxies]) / halo_mvir
            
            # Calculate statistics once for all arrays
            CentralHaloMass = np.log10(Mvir[w1])
            MeanCentralHaloMass.append(np.mean(CentralHaloMass))
            
            n_halos = len(BaryonFractions)
            sqrt_n = np.sqrt(n_halos)
            
            # Vectorized mean and std calculations
            means = [np.mean(arr) for arr in [BaryonFractions, StarsFractions, ColdFractions, 
                                             HotFractions, CGMFractions, ICSFractions, BHFractions, EjectedFractions]]
            stds = [np.std(arr) / sqrt_n for arr in [BaryonFractions, StarsFractions, ColdFractions, 
                                                    HotFractions, CGMFractions, ICSFractions, BHFractions, EjectedFractions]]
            
            # Append all means and bounds
            MeanBaryonFraction.append(means[0])
            MeanBaryonFractionU.append(means[0] + stds[0])
            MeanBaryonFractionL.append(means[0] - stds[0])
            
            MeanStars.append(means[1])
            MeanStarsU.append(means[1] + stds[1])
            MeanStarsL.append(means[1] - stds[1])
            
            MeanCold.append(means[2])
            MeanColdU.append(means[2] + stds[2])
            MeanColdL.append(means[2] - stds[2])
            
            MeanHot.append(means[3])
            MeanHotU.append(means[3] + stds[3])
            MeanHotL.append(means[3] - stds[3])
            
            MeanCGM.append(means[4])
            MeanCGMU.append(means[4] + stds[4])
            MeanCGML.append(means[4] - stds[4])
            
            MeanICS.append(means[5])
            MeanICSU.append(means[5] + stds[5])
            MeanICSL.append(means[5] - stds[5])
            
            MeanBH.append(means[6])
            MeanBHU.append(means[6] + stds[6])
            MeanBHL.append(means[6] - stds[6])

            MeanEjected.append(means[7])
            MeanEjectedU.append(means[7] + stds[7])
            MeanEjectedL.append(means[7] - stds[7])

    # Convert lists to arrays and ensure positive values for log scale
    MeanCentralHaloMass = np.array(MeanCentralHaloMass)
    MeanBaryonFraction = np.array(MeanBaryonFraction)
    MeanBaryonFractionU = np.array(MeanBaryonFractionU)
    MeanBaryonFractionL = np.maximum(np.array(MeanBaryonFractionL), 1e-6)  # Prevent negative values on log scale
    
    MeanStars = np.array(MeanStars)
    MeanStarsU = np.array(MeanStarsU)
    MeanStarsL = np.maximum(np.array(MeanStarsL), 1e-6)
    
    MeanCold = np.array(MeanCold)
    MeanColdU = np.array(MeanColdU)
    MeanColdL = np.maximum(np.array(MeanColdL), 1e-6)
    
    MeanHot = np.array(MeanHot)
    MeanHotU = np.array(MeanHotU)
    MeanHotL = np.maximum(np.array(MeanHotL), 1e-6)
    
    MeanCGM = np.array(MeanCGM)
    MeanCGMU = np.array(MeanCGMU)
    MeanCGML = np.maximum(np.array(MeanCGML), 1e-6)
    
    MeanICS = np.array(MeanICS)
    MeanICSU = np.array(MeanICSU)
    MeanICSL = np.maximum(np.array(MeanICSL), 1e-6)

    MeanBH = np.array(MeanBH)
    MeanBHU = np.array(MeanBHU)
    MeanBHL = np.maximum(np.array(MeanBHL), 1e-6)

    MeanEjected = np.array(MeanEjected)
    MeanEjectedU = np.array(MeanEjectedU)
    MeanEjectedL = np.maximum(np.array(MeanEjectedL), 1e-6)

    baryon_frac = 0.17
    plt.axhline(y=baryon_frac, color='grey', linestyle='--', linewidth=1.0, 
            label='Baryon Fraction = {:.2f}'.format(baryon_frac))

    # Add 1-sigma shading for each mass reservoir
    plt.fill_between(MeanCentralHaloMass, MeanBaryonFractionL, MeanBaryonFractionU, 
                     color='black', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanStarsL, MeanStarsU, 
                     color='purple', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanColdL, MeanColdU, 
                     color='blue', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanHotL, MeanHotU, 
                     color='red', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanCGML, MeanCGMU, 
                     color='green', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanICSL, MeanICSU, 
                     color='orange', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanEjectedL, MeanEjectedU, 
                     color='yellow', alpha=0.2)

    plt.plot(MeanCentralHaloMass, MeanBaryonFraction, 'k-', label='Total')
    plt.plot(MeanCentralHaloMass, MeanStars, label='Stars', color='purple', linestyle='--')
    plt.plot(MeanCentralHaloMass, MeanCold, label='Cold gas', color='blue', linestyle=':')
    plt.plot(MeanCentralHaloMass, MeanHot, label='Hot gas', color='red')
    plt.plot(MeanCentralHaloMass, MeanCGM, label='Circumgalactic Medium', color='green', linestyle='-.')
    plt.plot(MeanCentralHaloMass, MeanICS, label='Intracluster Stars', color='orange', linestyle='-.')
    plt.plot(MeanCentralHaloMass, MeanEjected, label='Ejected gas', color='yellow', linestyle='--')

    #plt.yscale('log')

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} \mathrm{Baryon\ Fraction}$')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.axis([11.1, 15.0, 0.0, 0.2])

    leg = plt.legend(loc='upper right', numpoints=1, labelspacing=0.1, bbox_to_anchor=(1.0, 0.5))
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    outputFile = OutputDir + '11.BaryonFraction' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()


# -------------------------------------------------------

    print('Plotting the mass in stellar, cold, hot, ejected, ICS reservoirs')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (Mvir > 1.0e10) & (StellarMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    HaloMass = np.log10(Mvir[w])
    plt.scatter(HaloMass, np.log10(StellarMass[w]), marker='o', s=0.3, c='k', alpha=0.5, label='Stars')
    plt.scatter(HaloMass, np.log10(ColdGas[w]), marker='o', s=0.3, color='blue', alpha=0.5, label='Cold gas')
    plt.scatter(HaloMass, np.log10(HotGas[w]), marker='o', s=0.3, color='red', alpha=0.5, label='Hot gas')
    plt.scatter(HaloMass, np.log10(EjectedMass[w]), marker='o', s=0.3, color='green', alpha=0.5, label='Ejected gas')
    plt.scatter(HaloMass, np.log10(IntraClusterStars[w]), marker='o', s=0.3, color='yellow', alpha=0.5, label='Intracluster stars')
    plt.scatter(HaloMass, np.log10(CGMgas[w]), marker='o', s=0.3, color='orange', alpha=0.5, label='CGM gas')

    plt.ylabel(r'$\mathrm{stellar,\ cold,\ hot,\ ejected,\ CGM,\ ICS\ mass}$')  # Set the y...
    plt.xlabel(r'$\log\ M_{\mathrm{vir}}\ (h^{-1}\ M_{\odot})$')  # and the x-axis labels
    
    plt.axis([10.0, 15.0, 7.5, 14.0])

    leg = plt.legend(loc='upper left')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
        
    outputFile = OutputDir + '12.MassReservoirScatter' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the spatial distribution of all galaxies')

    plt.figure()  # New figure

    w = np.where((Mvir > 0.0) & (StellarMass > 1.0e9))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    xx = Posx[w]
    yy = Posy[w]
    zz = Posz[w]

    buff = BoxSize*0.1

    ax = plt.subplot(221)  # 1 plot on the figure
    plt.scatter(xx, yy, marker='o', s=0.3, c='k', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')  # Set the y...
    plt.xlabel(r'$\mathrm{y}$')  # and the x-axis labels
    
    ax = plt.subplot(222)  # 1 plot on the figure
    plt.scatter(xx, zz, marker='o', s=0.3, c='k', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')  # Set the y...
    plt.xlabel(r'$\mathrm{z}$')  # and the x-axis labels
    
    ax = plt.subplot(223)  # 1 plot on the figure
    plt.scatter(yy, zz, marker='o', s=0.3, c='k', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{y}$')  # Set the y...
    plt.xlabel(r'$\mathrm{z}$')  # and the x-axis labels
        
    outputFile = OutputDir + '13.SpatialDistribution' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the SFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w2 = np.where(StellarMass > 0.01)[0]
    if(len(w2) > dilute): w2 = sample(list(range(len(w2))), dilute)
    mass = np.log10(StellarMass[w2])
    starformationrate =  (SfrDisk[w2] + SfrBulge[w2])

    # Create scatter plot with metallicity coloring
    plt.scatter(mass, np.log10(starformationrate), c='b', marker='o', s=1, alpha=0.7)

    plt.ylabel(r'$\log_{10} \mathrm{SFR}\ (M_{\odot}\ \mathrm{yr^{-1}})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.xlim(6.0, 12.2)
    plt.ylim(-5, 3)  # Set y-axis limits for SFR

    outputFile = OutputDir + '14.StarFormationRate' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting H2 surface density vs SFR surface density')

    plt.figure()  # New figure
    # Σ_H2 in M_sun/pc^2, Σ_SFR in M_sun/yr/kpc^2

    sfrdot = SfrDisk + SfrBulge
    Mvir = read_hdf(snap_num = Snapshot, param = 'Mvir') * 1.0e10 / Hubble_h
    H2Gas = read_hdf(snap_num = Snapshot, param = 'H2gas') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(snap_num = Snapshot, param = 'DiskRadius')  # in Mpc/h

    w = np.where((StellarMass > 0.0) & (H2Gas > 0.0) & (sfrdot > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    disk_radius = DiskRadius[w] * 1.0e6 / Hubble_h
    disk_area = 2 * np.pi * disk_radius**2

    sigma_H2 = H2Gas[w] / disk_area # DiskRadius in kpc, area in pc^2
    sigma_SFR = sfrdot[w] / disk_area * 1.0e6 # area in kpc^2
    log10_sigma_H2 = np.log10(sigma_H2)
    log10_sigma_SFR = np.log10(sigma_SFR)
    # Color by Mvir (virial mass)
    sc = plt.scatter(log10_sigma_H2, log10_sigma_SFR, c=np.log10(StellarMass[w]), cmap='plasma',
                      alpha=0.6, s=5, vmin=8, vmax=12, label='SAGE25')
    cb = plt.colorbar(sc)

    cb.set_label(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    # Add canonical Kennicutt-Schmidt law (Kennicutt 1998): log(Sigma_SFR) = 1.4*log(Sigma_gas) - 3.6
    sigma_gas_range = np.linspace(-4, 4, 100)
    ks_law = 1.4 * sigma_gas_range - 3.6
    plt.plot(sigma_gas_range, ks_law, linestyle='--', color='red', label='Kennicutt (1998)')

    gas_range = np.logspace(-4, 4, 100)
        
    # Bigiel et al. (2008) - resolved regions in nearby galaxies
    # Σ_SFR = 1.6e-3 × (Σ_H2)^1.0
    ks_bigiel = np.log10(1.6e-3) + 1.0 * np.log10(gas_range)
    plt.plot(np.log10(gas_range), ks_bigiel, linestyle=':', color='red', linewidth=2.5, alpha=0.8, 
            label='Bigiel+ (2008) - resolved', zorder=2)
    
    # Schruba et al. (2011) - different normalization
    # Σ_SFR = 2.1e-3 × (Σ_H2)^1.0
    ks_schruba = np.log10(2.1e-3) + 1.0 * np.log10(gas_range)
    plt.plot(np.log10(gas_range), ks_schruba, linestyle='--', color='red', linewidth=2, alpha=0.6, 
            label='Schruba+ (2011)', zorder=2)
            
    # Leroy et al. (2013) - whole galaxy integrated
    # Σ_SFR = 1.4e-3 × (Σ_H2)^1.1
    ks_leroy = np.log10(1.4e-3) + 1.1 * np.log10(gas_range)
    plt.plot(np.log10(gas_range), ks_leroy, linestyle='-', color='red', linewidth=2, alpha=0.7, 
            label='Leroy+ (2013) - galaxies', zorder=2)
            
    # Saintonge et al. (2011) - COLD GASS survey
    # Σ_SFR = 1.0e-3 × (Σ_H2)^0.96
    ks_saintonge = np.log10(1.0e-3) + 0.96 * np.log10(gas_range)
    plt.plot(np.log10(gas_range), ks_saintonge, linestyle='-.', color='red', linewidth=1.5, alpha=0.5, 
            label='Saintonge+ (2011)', zorder=2)
    
    plt.xlabel(r'$\log_{10} \Sigma_{\mathrm{H}_2}\ (M_{\odot}/\mathrm{pc}^2)$')
    plt.ylabel(r'$\log_{10} \Sigma_{\mathrm{SFR}}\ (M_{\odot}/yr/\mathrm{kpc}^2)$')
    # # plt.title('H$_2$ Surface Density vs SFR Surface Density (K-S Law)')
    plt.legend(loc='lower right', fontsize='small', frameon=False)
    plt.xlim(-3, 4)
    plt.ylim(-6, 1)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outputFile = OutputDir + '15.h2_vs_sfr_surface_density' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()


    # -------------------------------------------------------

    print('Plotting Size-Mass relation split by star-forming and quiescent')

    plt.figure()  # New figure

    w = np.where((Mvir > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_disk_radius = np.log10(DiskRadius[w] / 0.001)
    log10_disk_radius_quiescent = np.log10(DiskRadius[w] / 0.001 / 1.67)
    SFR = SfrDisk[w] + SfrBulge[w]
    sSFR = np.full_like(SFR, -99.0)
    mask = (StellarMass[w] > 0)
    sSFR[mask] = np.log10(SFR[mask] / StellarMass[w][mask])

    star_forming = sSFR > sSFRcut
    quiescent = sSFR <= sSFRcut


    plt.scatter(log10_stellar_mass[star_forming], log10_disk_radius[star_forming], c='darkblue', s=5, alpha=0.1)
    plt.scatter(log10_stellar_mass[quiescent], log10_disk_radius_quiescent[quiescent], c='darkred', s=5, alpha=0.1)

    # Add median lines for both populations

    def median_and_sigma(x, y, bins):
        bin_centers = []
        medians = []
        sig_low = []
        sig_high = []
        for i in range(len(bins)-1):
            mask = (x >= bins[i]) & (x < bins[i+1])
            if np.any(mask):
                bin_centers.append(0.5*(bins[i]+bins[i+1]))
                medians.append(np.median(y[mask]))
                sig_low.append(np.percentile(y[mask], 16))
                sig_high.append(np.percentile(y[mask], 84))
        return (np.array(bin_centers), np.array(medians), np.array(sig_low), np.array(sig_high))

    bins = np.arange(6, 12.1, 0.3)

    # Star-forming median and 1-sigma
    x_sf, y_sf, y_sf_low, y_sf_high = median_and_sigma(log10_stellar_mass[star_forming], log10_disk_radius[star_forming], bins)
    plt.plot(x_sf, y_sf, c='darkblue', lw=2.5, label='Median SF')
    plt.fill_between(x_sf, y_sf_low, y_sf_high, color='darkblue', alpha=0.18, label='SF 1$\sigma$')
    # Quiescent median and 1-sigma
    x_q, y_q, y_q_low, y_q_high = median_and_sigma(log10_stellar_mass[quiescent], log10_disk_radius_quiescent[quiescent], bins)
    plt.plot(x_q, y_q, c='darkred', lw=2.5, label='Median Q')
    plt.fill_between(x_q, y_q_low, y_q_high, color='darkred', alpha=0.18, label='Q 1$\sigma$')

    # Approximate Shen+2003 relation
    M_star = np.logspace(6, 12, 100)
    R_shen_sf = 3.0 * (M_star/1e10)**0.14  # Star-forming
    R_shen_q = 1.5 * (M_star/1e10)**0.12   # Quiescent (smaller, shallower)

    plt.plot(np.log10(M_star), np.log10(R_shen_sf), 'b-', linewidth=2, label='Shen+03 SF')
    plt.plot(np.log10(M_star), np.log10(R_shen_q), 'r-', linewidth=2, label='Shen+03 Q')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} R_{\mathrm{disk}}\ (\mathrm{kpc})$')
    # # plt.title('Size-Mass Relation: Star Forming (blue) vs Quiescent (red)')
    plt.legend(loc='upper left', fontsize='small', frameon=False)
    plt.xlim(6, 12)
    plt.ylim(-1, 2.5)

    outputFile = OutputDir + '16.size_mass_relation_split' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting CGM gas fraction vs stellar mass')

    plt.figure()

    w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])

    # Calculate total hot-type gas and CGM fraction
    total_hot_gas = EjectedMass[w] + HotGas[w]
    # Avoid division by zero
    mask = total_hot_gas > 0
    f_CGM = np.zeros_like(total_hot_gas)
    f_CGM[mask] = EjectedMass[w][mask] / total_hot_gas[mask]

    # Only plot where there's actually gas
    valid = mask & (f_CGM >= 0) & (f_CGM <= 1)

    plt.scatter(log10_stellar_mass[valid], f_CGM[valid], c='purple', s=5, alpha=0.6)

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$f_{\mathrm{CGM}} = M_{\mathrm{CGM}}/(M_{\mathrm{CGM}} + M_{\mathrm{hot}})$')
    plt.xlim(8, 12)
    plt.ylim(0, 1)

    outputFile = OutputDir + '17.cgm_gas_fraction' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting Black Hole Mass vs Stellar Mass')

    # In your plotting script
    plt.figure()
    w = np.where((Mvir > 0.0) & (StellarMass > 0.0) & (BlackHoleMass > 0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_BH_mass = np.log10(BlackHoleMass[w])

    # Calculate total hot-type gas and CGM fraction
    total_hot_gas = EjectedMass[w] + HotGas[w]
    # Avoid division by zero
    mask = total_hot_gas > 0
    f_CGM = np.zeros_like(total_hot_gas)
    f_CGM[mask] = EjectedMass[w][mask] / total_hot_gas[mask]

    # Only plot where there's actually gas
    valid = mask & (f_CGM >= 0) & (f_CGM <= 1)

    plt.scatter(log10_stellar_mass[valid], log10_BH_mass[valid], c=f_CGM[valid], s=5, cmap='plasma')
    plt.colorbar(label='f_CGM')
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{BH}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(4, 10)

    outputFile = OutputDir + '18.BH_mass_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

        # -------------------------------------------------------

    print('Plotting CGM vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (CGMgas > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_CGM_mass = np.log10(CGMgas[w])
    tvir = np.log10(35.9 * Vvir[w]**2)  # in Kelvin

    data = """10.06276150627615, 10.48936170212766
        10.112970711297072, 10.510638297872342
        10.175732217573222, 10.531914893617023
        10.242677824267782, 10.574468085106384
        10.322175732217573, 10.617021276595747
        10.401673640167363, 10.680851063829788
        10.481171548117155, 10.702127659574469
        10.560669456066945, 10.74468085106383
        10.644351464435147, 10.765957446808512
        10.719665271966527, 10.829787234042554
        10.786610878661088, 10.872340425531917
        10.866108786610878, 10.914893617021278
        10.94560669456067, 10.97872340425532
        11.02510460251046, 11.085106382978724
        11.108786610878662, 11.127659574468087
        11.196652719665272, 11.297872340425533
        11.276150627615063, 11.425531914893618
        11.359832635983263, 11.574468085106384
        11.426778242677823, 11.765957446808512
        11.497907949790795, 11.936170212765958
        11.581589958158995, 12.106382978723406
        11.652719665271967, 12.255319148936172
        11.728033472803347, 12.340425531914896
        11.782426778242678, 12.425531914893618
        11.832635983263598, 12.468085106382981
        11.870292887029288, 12.638297872340427
        11.912133891213388, 12.787234042553193
        11.94979079497908, 12.893617021276597
        12, 12.829787234042556
        12.05020920502092, 12.808510638297875
        12.09623430962343, 12.872340425531917
        12.138075313807532, 12.95744680851064
        12.184100418410042, 13.085106382978726"""

    # Split the data into lines and extract x, y coordinates
    lines = data.strip().split('\n')
    x = []
    y = []

    for line in lines:
        coords = line.split(', ')
        x.append(float(coords[0]))
        y.append(float(coords[1]))

    # Convert to numpy arrays (optional, but often useful for plotting)
    tng_x = np.array(x)
    tng_y = np.array(y)

    plt.scatter(log10_stellar_mass, log10_CGM_mass, c=Vvir[w], cmap='seismic', s=5)
    plt.plot(tng_x, tng_y, 'k--', lw=2, label='TNG-Cluster')
    plt.colorbar(label=r'$V_{\mathrm{vir}}\ (\mathrm{km/s})$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{CGM}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '19.CGM_mass_vs_stellar_mass_temperature' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

        # -------------------------------------------------------

    print('Plotting CGM vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (CGMgas > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_CGM_mass = np.log10(CGMgas[w])
    Z = np.log10((MetalsCGMgas[w] / CGMgas[w]) / 0.02) + 9.0

    data = """10.06276150627615, 10.48936170212766
        10.112970711297072, 10.510638297872342
        10.175732217573222, 10.531914893617023
        10.242677824267782, 10.574468085106384
        10.322175732217573, 10.617021276595747
        10.401673640167363, 10.680851063829788
        10.481171548117155, 10.702127659574469
        10.560669456066945, 10.74468085106383
        10.644351464435147, 10.765957446808512
        10.719665271966527, 10.829787234042554
        10.786610878661088, 10.872340425531917
        10.866108786610878, 10.914893617021278
        10.94560669456067, 10.97872340425532
        11.02510460251046, 11.085106382978724
        11.108786610878662, 11.127659574468087
        11.196652719665272, 11.297872340425533
        11.276150627615063, 11.425531914893618
        11.359832635983263, 11.574468085106384
        11.426778242677823, 11.765957446808512
        11.497907949790795, 11.936170212765958
        11.581589958158995, 12.106382978723406
        11.652719665271967, 12.255319148936172
        11.728033472803347, 12.340425531914896
        11.782426778242678, 12.425531914893618
        11.832635983263598, 12.468085106382981
        11.870292887029288, 12.638297872340427
        11.912133891213388, 12.787234042553193
        11.94979079497908, 12.893617021276597
        12, 12.829787234042556
        12.05020920502092, 12.808510638297875
        12.09623430962343, 12.872340425531917
        12.138075313807532, 12.95744680851064
        12.184100418410042, 13.085106382978726"""

    # Split the data into lines and extract x, y coordinates
    lines = data.strip().split('\n')
    x = []
    y = []

    for line in lines:
        coords = line.split(', ')
        x.append(float(coords[0]))
        y.append(float(coords[1]))

    # Convert to numpy arrays (optional, but often useful for plotting)
    tng_x = np.array(x)
    tng_y = np.array(y)

    plt.scatter(log10_stellar_mass, log10_CGM_mass, c=Z, cmap='plasma', s=5, vmin=7, vmax=9)
    plt.plot(tng_x, tng_y, 'k--', lw=2, label='TNG-Cluster')
    plt.colorbar(label=r'$12\ +\ \log_{10}[\mathrm{O/H}]$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{CGM}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '20.CGM_mass_vs_stellar_mass_metallicity' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting Ejected vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (EjectedMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_CGM_mass = np.log10(EjectedMass[w])
    tvir = np.log10(35.9 * Vvir[w]**2)  # in Kelvin

    plt.scatter(log10_stellar_mass, log10_CGM_mass, c=Vvir[w], cmap='seismic', s=5)
    plt.colorbar(label=r'$V_{\mathrm{vir}}\ (\mathrm{km/s})$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{ejected}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '19.Ejected_mass_vs_stellar_mass_temperature' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting Ejected vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (EjectedMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_ejected_mass = np.log10(EjectedMass[w])
    Z = np.log10((MetalsEjectedMass[w] / EjectedMass[w]) / 0.02) + 9.0

    plt.scatter(log10_stellar_mass, log10_ejected_mass, c=Z, cmap='plasma', s=5, vmin=7, vmax=9)
    plt.colorbar(label=r'$12\ +\ \log_{10}[\mathrm{O/H}]$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{ejected}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '20.Ejected_mass_vs_stellar_mass_metallicity' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting Hot gas vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (HotGas > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    log10_CGM_mass = np.log10(HotGas[w])
    Z = np.log10((MetalsHotGas[w] / HotGas[w]) / 0.02) + 9.0

    plt.scatter(log10_stellar_mass, log10_CGM_mass, c=Z, cmap='plasma', s=5, vmin=7, vmax=9)
    plt.colorbar(label=r'$12\ +\ \log_{10}[\mathrm{O/H}]$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{hot}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '20.Hot_gas_vs_stellar_mass_metallicity' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting ICS vs Hot Gas Mass')

    plt.figure()
    w = np.where((IntraClusterStars > 0.0) & (CGMgas > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(CGMgas[w] + HotGas[w])
    log10_CGM_mass = np.log10(IntraClusterStars[w])
    Z = np.log10((MetalsCGMgas[w] + MetalsHotGas[w]) / (CGMgas[w] + HotGas[w]) / 0.02) + 9.0

    plt.scatter(log10_stellar_mass, log10_CGM_mass, c=Z, cmap='plasma', s=5, vmin=7, vmax=9)
    plt.colorbar(label=r'$12\ +\ \log_{10}[\mathrm{O/H}]$')

    plt.xlabel(r'$\log_{10} M_{\mathrm{CGM\ +\ hot}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{ICS}}\ (M_{\odot})$')

    plt.xlim(8, 12)
    plt.ylim(6, 12)

    outputFile = OutputDir + '20.ICS_vs_hot_gas_mass_metallicity' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    print('Plotting outflow vs Vvir')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (OutflowRate > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    mass_loading = OutflowRate[w]

    plt.scatter(Vvir[w], mass_loading, c='green', s=5, alpha=0.6)

    plt.xlabel(r'$V_{\mathrm{vir}}\ (\mathrm{km/s})$')
    plt.ylabel(r'$\dot{M}_{\mathrm{outflow}}\ (M_{\odot}\ \mathrm{yr}^{-1})$')

    # Add vertical line at critical velocity
    plt.axvline(x=60, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
                label='$V_{\\mathrm{crit}} = 60$ km/s')

    plt.xlim(min(Vvir[w]), 300)
    plt.ylim(0.01, max(mass_loading)*1.1)

    outputFile = OutputDir + '21.outflow_rate_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # -------------------------------------------------------

    # Regime = read_hdf(snap_num = Snapshot, param = 'Regime')

    print('Regime fractions:')
    print('Cool regime:', np.mean(Regime == 0))
    print('Hot regime:', np.mean(Regime == 1))

    print('Plotting stellar mass vs halo mass colored by regime')

    plt.figure()

    w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_halo_mass = np.log10(Mvir[w])
    log10_stellar_mass = np.log10(StellarMass[w])
    regime_values = Regime[w]
    
    # Separate the data by regime for different colors
    cgm_regime = (regime_values == 0)
    hot_regime = (regime_values == 1)
    
    # Plot each regime separately with different colors
    if np.any(cgm_regime):
        plt.scatter(log10_halo_mass[cgm_regime], log10_stellar_mass[cgm_regime], 
                   c='blue', s=5, alpha=0.6, label='CGM Regime')
    
    if np.any(hot_regime):
        plt.scatter(log10_halo_mass[hot_regime], log10_stellar_mass[hot_regime], 
                   c='red', s=5, alpha=0.6, label='Hot Regime')

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.xlim(10, 15)
    plt.ylim(8, 12)
    
    # Add legend
    plt.legend(loc='upper left', frameon=False)

    outputFile = OutputDir + '22.stellar_vs_halo_mass_by_regime' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting specific SFR vs stellar mass')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    dilute = 100000

    w2 = np.where(StellarMass > 0.0)[0]
    if(len(w2) > dilute): w2 = sample(list(range(len(w2))), dilute)
    mass = np.log10(StellarMass[w2])
    starformationrate = (SfrDisk[w2] + SfrBulge[w2])
    sSFR = np.full_like(starformationrate, -99.0)
    mask = (StellarMass[w2] > 0)
    sSFR[mask] = np.log10(starformationrate[mask] / StellarMass[w2][mask])

    sSFRcut = -11.0
    print(f'sSFR cut at {sSFRcut} yr^-1')

    # Separate populations
    sf_mask = (sSFR > sSFRcut) & (sSFR > -99.0)  # Star-forming
    q_mask = (sSFR <= sSFRcut) & (sSFR > -99.0)  # Quiescent

    mass_sf = mass[sf_mask]
    sSFR_sf = sSFR[sf_mask]
    mass_q = mass[q_mask]
    sSFR_q = sSFR[q_mask]

    # Define grid for density calculation
    x_bins = np.linspace(8.0, 12.2, 100)
    y_bins = np.linspace(-13, -8, 100)

    def plot_density_contours(x, y, color, label, clip_above=None, clip_below=None):
        """Plot filled contours with 1, 2, 3 sigma levels"""
        if len(x) < 10:
            return
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H = H.T  # Transpose for correct orientation
        
        # Smooth the histogram
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=1.5)
        
        # Apply clipping if specified
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        if clip_above is not None:
            # Mask out regions above the clip line
            mask_2d = y_centers[:, np.newaxis] <= clip_above
            H_smooth = H_smooth * mask_2d
        if clip_below is not None:
            # Mask out regions below the clip line
            mask_2d = y_centers[:, np.newaxis] >= clip_below
            H_smooth = H_smooth * mask_2d
        
        # Calculate contour levels
        sorted_H = np.sort(H_smooth.flatten())[::-1]
        sorted_H = sorted_H[sorted_H > 0]  # Remove zeros
        if len(sorted_H) == 0:
            return
            
        cumsum = np.cumsum(sorted_H)
        cumsum = cumsum / cumsum[-1]
        
        level_3sigma = sorted_H[np.where(cumsum >= 0.997)[0][0]] if np.any(cumsum >= 0.997) else sorted_H[-1]
        level_2sigma = sorted_H[np.where(cumsum >= 0.95)[0][0]] if np.any(cumsum >= 0.95) else sorted_H[-1]
        level_1sigma = sorted_H[np.where(cumsum >= 0.68)[0][0]] if np.any(cumsum >= 0.68) else sorted_H[-1]
        
        levels = [level_3sigma, level_2sigma, level_1sigma]
        alphas = [0.3, 0.5, 0.7]
        
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        
        # Plot filled contours
        for i, (level, alpha) in enumerate(zip(levels, alphas)):
            if i == len(levels) - 1:
                ax.contourf(x_centers, y_centers, H_smooth, 
                        levels=[level, H_smooth.max()],
                        colors=[color], alpha=alpha, label=label)
            else:
                ax.contourf(x_centers, y_centers, H_smooth, 
                        levels=[level, levels[i+1] if i+1 < len(levels) else H_smooth.max()],
                        colors=[color], alpha=alpha)
        
        # Add contour lines
        ax.contour(x_centers, y_centers, H_smooth, 
                levels=levels, colors=color, linewidths=1.0, alpha=0.8)

    # Plot quiescent population (red) - clip above -11
    if len(mass_q) > 0:
        plot_density_contours(mass_q, sSFR_q, 'firebrick', 'Quiescent', clip_above=sSFRcut)

    # Plot star-forming population (blue) - clip below -11
    if len(mass_sf) > 0:
        plot_density_contours(mass_sf, sSFR_sf, 'dodgerblue', 'Star-forming', clip_below=sSFRcut)
    # Add the sSFR cut line
    plt.axhline(y=sSFRcut, color='black', linestyle='--', linewidth=2, 
            label=f'sSFR cut = {sSFRcut}', zorder=10)

    plt.ylabel(r'$\log_{10} \mathrm{sSFR}\ (\mathrm{yr^{-1}})$', fontsize=14)
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$', fontsize=14)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.xlim(8.0, 12.2)
    plt.ylim(-13, -8)

    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(OutputDir + '23.specific_star_formation_rate' + OutputFormat, dpi=150)
    print('Saved to', OutputDir + '23.specific_star_formation_rate' + OutputFormat, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Mass loading factor statistics:')
    print('Mean:', np.mean(MassLoading))
    print('Median:', np.median(MassLoading))
    print('Std Dev:', np.std(MassLoading))
    print('Max:', np.max(MassLoading))
    print('Min:', np.min(MassLoading))
    print('Sample of values:', MassLoading[:10])

    print('Plotting Mass Loading Factor vs Stellar Mass')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (MassLoading >= 0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    MassLoading = MassLoading[w]

    plt.scatter(log10_stellar_mass, MassLoading, c='b', marker='o', s=1, alpha=0.7)
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\mathrm{Mass\ Loading\ Factor}\ (\eta)$')
    plt.xlim(8.0, 12.2)
    plt.ylim(0, None)

    plt.savefig(OutputDir + '24.mass_loading_factor_vs_stellar_mass' + OutputFormat)
    print('Saved to', OutputDir + '24.mass_loading_factor_vs_stellar_mass' + OutputFormat, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting Cooling Rate vs. Temperature')

    plt.figure()

    print('Cooling Rate statistics:')
    print('Mean:', np.mean(Cooling))
    print('Median:', np.median(Cooling))
    print('Std Dev:', np.std(Cooling))
    print('Max:', np.max(Cooling))
    print('Min:', np.min(Cooling))
    print('Sample of values:', Cooling[:10])

    print('Temperature statistics:')
    print('Mean:', np.mean(Tvir))
    print('Median:', np.median(Tvir))
    print('Std Dev:', np.std(Tvir))
    print('Max:', np.max(Tvir))
    print('Min:', np.min(Tvir))
    print('Sample of values:', Tvir[:10])

    # Filter for valid cooling and temperature
    w = np.where((Cooling > 0) & (Tvir > 0))[0]

    # Convert Temperature to keV and take log10
    log_T_keV = np.log10(Tvir[w] * 8.6e-8)

    # Convert Cooling to units of 10^40 erg/s (log scale)
    # Cooling is already log10(erg/s), so we subtract 40
    log_Cooling_40 = Cooling[w] - 40.0

    plt.scatter(log_T_keV, log_Cooling_40, c='grey', marker='x', s=50, alpha=0.3)
    plt.xlabel(r'$\log_{10} T_{\mathrm{vir}}\ [\mathrm{keV}]$')
    plt.ylabel(r'$\log_{10} \mathrm{Net\ Cooling}\ [10^{40}\ \mathrm{erg\ s^{-1}}]$')

    plt.xlim(-0.2, 1.0)
    plt.ylim(-1.0, 6.0)

    plt.savefig(OutputDir + '25.cooling_rate_vs_temperature' + OutputFormat)
    print('Saved to', OutputDir + '25.cooling_rate_vs_temperature' + OutputFormat, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting Bulge Size vs Bulge Mass')

    plt.figure()
    w = np.where((BulgeMass > 0.0) & (BulgeRadius > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_bulge_mass = np.log10(BulgeMass[w])
    log10_bulge_radius = np.log10(BulgeRadius[w] / 0.001)  # Convert to kpc
    bulge_fraction = BulgeMass[w] / StellarMass[w]

    # Color by bulge fraction
    sc = plt.scatter(log10_bulge_mass, log10_bulge_radius, c=bulge_fraction, 
                    cmap='RdYlBu_r', s=5, alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(sc, label=r'$f_{\mathrm{bulge}} = M_{\mathrm{bulge}}/M_{\mathrm{stars}}$')

    # Add the theoretical mass-size relation
    # R_e = 3.5 kpc * (M_bulge / 10^11 Msun)^0.55 (Shen+2003, offset for bulges per Gadotti 2009)
    M_bulge_range = np.logspace(8, 12, 100)
    R_bulge_theory = 3.5 * (M_bulge_range / 1e11)**0.55
    plt.plot(np.log10(M_bulge_range), np.log10(R_bulge_theory), 
            'k--', linewidth=2, label=r'$R_e = 3.5(M/10^{11})^{0.55}$ kpc', zorder=10)

    # Add text annotation
    plt.text(11.0, 1.5, 'Shen+03 (early-type)\nGadotti 09 (bulges)', 
            fontsize=10, color='black', alpha=0.7)

    plt.xlabel(r'$\log_{10} M_{\mathrm{bulge}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} R_{\mathrm{bulge}}\ (\mathrm{kpc})$')
    plt.xlim(8, 12)
    plt.ylim(-0.5, 2.0)
    plt.legend(loc='upper left', frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '26.bulge_size_mass_relation' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting Bulge vs Disk Size')

    plt.figure()
    w = np.where((BulgeMass > 0.0) & (BulgeRadius > 0.0) & (DiskRadius > 0.0) & 
                (StellarMass > 1e9))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_disk_radius = np.log10(DiskRadius[w] / 0.001)  # Convert to kpc
    log10_bulge_radius = np.log10(BulgeRadius[w] / 0.001)  # Convert to kpc
    log10_stellar_mass = np.log10(StellarMass[w])

    # Color by total stellar mass
    # sc = plt.scatter(log10_disk_radius, log10_bulge_radius, c=log10_stellar_mass,
    #                 cmap='plasma', s=5, alpha=0.6, vmin=9, vmax=12)
    # plt.colorbar(sc, label=r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')

    # Add Merger Bulge Radius
    w_merger = np.where((MergerBulgeRadius > 0.0) & (DiskRadius > 0.0) & (StellarMass > 1e9))[0]
    if(len(w_merger) > dilute): w_merger = sample(list(w_merger), dilute)

    log10_merger_radius = np.log10(MergerBulgeRadius[w_merger] / 0.001)
    log10_disk_radius_merger = np.log10(DiskRadius[w_merger] / 0.001)
    log10_stellar_mass = np.log10(StellarMass[w_merger])
    sc = plt.scatter(log10_disk_radius_merger, log10_merger_radius, c=log10_stellar_mass, marker='d', edgecolors='k',
                    cmap='plasma', s=50, alpha=0.4, label='Merger Bulge')

    # Add Instability Bulge Radius
    w_instab = np.where((InstabilityBulgeRadius > 0.0) & (DiskRadius > 0.0) & (StellarMass > 1e9))[0]
    if(len(w_instab) > dilute): w_instab = sample(list(w_instab), dilute)
    log10_instab_radius = np.log10(InstabilityBulgeRadius[w_instab] / 0.001)
    log10_disk_radius_instab = np.log10(DiskRadius[w_instab] / 0.001)
    log10_stellar_mass = np.log10(StellarMass[w_instab])
    plt.scatter(log10_disk_radius_instab, log10_instab_radius, c=log10_stellar_mass, marker='s', 
                    cmap='plasma', s=5, alpha=0.4, label='Instability Bulge')

    plt.colorbar(sc, label=r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')

    # Add 1:1 line
    plt.plot([-1, 3], [-1, 3], 'k:', linewidth=1, alpha=0.5, label='1:1')

    # Add typical ratio line (bulge ~ 0.1 * disk)
    disk_range = np.linspace(-1, 3, 100)
    plt.plot(disk_range, disk_range + np.log10(0.1), 'r--', 
            linewidth=2, label=r'$R_{\mathrm{bulge}} = 0.1 R_{\mathrm{disk}}$', alpha=0.7)

    plt.xlabel(r'$\log_{10} R_{\mathrm{disk}}\ (\mathrm{kpc})$')
    plt.ylabel(r'$\log_{10} R_{\mathrm{bulge}}\ (\mathrm{kpc})$')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-1.5, 2.0)
    plt.legend(loc='upper left', frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '27.bulge_vs_disk_size' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()
    # -------------------------------------------------------

    print('Plotting Mass Components vs Stellar Mass')

    plt.figure()
    w = np.where(StellarMass > 1e9)[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    disk_mass = StellarMass[w] - BulgeMass[w]
    # Ensure positive values for log
    disk_mass[disk_mass <= 0] = 1e-10
    merger_bulge_mass = MergerBulgeMass[w]
    merger_bulge_mass[merger_bulge_mass <= 0] = 1e-10
    instability_bulge_mass = InstabilityBulgeMass[w]
    instability_bulge_mass[instability_bulge_mass <= 0] = 1e-10

    plt.scatter(log10_stellar_mass, np.log10(disk_mass), c='b', s=10, alpha=0.8, label='Disk Mass', marker='s')
    plt.scatter(log10_stellar_mass, np.log10(merger_bulge_mass), c='r', s=10, alpha=0.6, label='Merger Bulge Mass', marker='s')
    plt.scatter(log10_stellar_mass, np.log10(instability_bulge_mass), c='greenyellow', s=10, alpha=0.3, label='Instability Bulge Mass', marker='s')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{component}}\ (M_{\odot})$')
    plt.xlim(9, 12)
    plt.ylim(8, 12)
    plt.legend(loc='upper left', frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '28.mass_components_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting Mass Ratios vs Stellar Mass')

    plt.figure()
    w = np.where(StellarMass > 1e9)[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    disk_ratio = (StellarMass[w] - BulgeMass[w]) / StellarMass[w]
    merger_ratio = MergerBulgeMass[w] / StellarMass[w]
    instability_ratio = InstabilityBulgeMass[w] / StellarMass[w]

    plt.scatter(log10_stellar_mass, disk_ratio, c='b', s=10, alpha=0.8, label='Disk Fraction', marker='s')
    plt.scatter(log10_stellar_mass, merger_ratio, c='r', s=10, alpha=0.6, label='Merger Bulge Fraction', marker='s')
    plt.scatter(log10_stellar_mass, instability_ratio, c='greenyellow', s=10, alpha=0.3, label='Instability Bulge Fraction', marker='s')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'Mass Fraction')
    plt.xlim(9, 12)
    plt.ylim(0, 1.05)
    # plt.legend(loc='center left', frameon=False)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '29.mass_ratios_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()
    
    # -------------------------------------------------------

    print('Plotting stellar to halo mass relation')

    plt.figure()
    w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_halo_mass = np.log10(Mvir[w])
    log10_stellar_mass = np.log10(StellarMass[w])

    plt.scatter(log10_halo_mass, log10_stellar_mass, c='dodgerblue', s=5, alpha=0.2)

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.xlim(10, 15)
    plt.ylim(7, 12)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '30.stellar_to_halo_mass_relation' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting stellar to halo mass ratio')

    plt.figure()
    w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]

    log10_halo_mass = np.log10(Mvir[w])
    stellar_to_halo_ratio = StellarMass[w] / Mvir[w]

    # Create halo mass bins
    halo_mass_bins = np.arange(10.0, 15.5, 0.2)
    bin_centers = (halo_mass_bins[:-1] + halo_mass_bins[1:]) / 2
    
    # Calculate median and bootstrap errors for each bin
    n_bootstrap = 1000
    medians = []
    lower_errors = []
    upper_errors = []
    
    for i in range(len(halo_mass_bins) - 1):
        mask = (log10_halo_mass >= halo_mass_bins[i]) & (log10_halo_mass < halo_mass_bins[i+1])
        bin_data = stellar_to_halo_ratio[mask]
        
        if len(bin_data) > 10:  # Only calculate if we have enough data points
            median = np.median(bin_data)
            medians.append(median)
            
            # Bootstrap to estimate errors
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                resample = np.random.choice(bin_data, size=len(bin_data), replace=True)
                bootstrap_medians.append(np.median(resample))
            
            # 16th and 84th percentiles for 1-sigma equivalent errors
            lower = np.percentile(bootstrap_medians, 16)
            upper = np.percentile(bootstrap_medians, 84)
            lower_errors.append(lower)
            upper_errors.append(upper)
        else:
            medians.append(np.nan)
            lower_errors.append(np.nan)
            upper_errors.append(np.nan)
    
    medians = np.array(medians)
    lower_errors = np.array(lower_errors)
    upper_errors = np.array(upper_errors)
    
    # Remove NaN values
    valid = ~np.isnan(medians)
    bin_centers_valid = bin_centers[valid]
    medians_valid = medians[valid]
    lower_errors_valid = lower_errors[valid]
    upper_errors_valid = upper_errors[valid]
    
    # Plot median line
    plt.plot(bin_centers_valid, medians_valid, color='dodgerblue', linewidth=2, label='Median')
    
    # Plot shaded area for bootstrap errors
    plt.fill_between(bin_centers_valid, lower_errors_valid, upper_errors_valid, 
                     color='dodgerblue', alpha=0.3, label='Bootstrap 1σ')

    # Load and plot observational data
    import pandas as pd
    obs_data = pd.read_csv('./data/SHMratio_data.csv', skiprows=1)
    
    # Each pair of columns is X and Y
    for i in range(0, len(obs_data.columns), 2):
        x_col = obs_data.iloc[:, i]
        y_col = obs_data.iloc[:, i+1]
        
        # Remove NaN values
        valid_obs = ~(pd.isna(x_col) | pd.isna(y_col))
        x_valid = x_col[valid_obs].values
        y_valid = 10**y_col[valid_obs].values  # Convert from log10 to linear
        
        plt.plot(x_valid, y_valid, linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$M_{\mathrm{stars}} / M_{\mathrm{vir}}$')
    plt.xlim(10, 15)
    # plt.ylim(0, 0.5)
    plt.yscale('log')
    plt.legend(loc='best', frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '31.stellar_to_halo_mass_ratio' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting central stellar density within 1 kpc vs stellar mass')

    plt.figure()
    ax = plt.subplot(111)

    # Calculate disk mass (total stellar mass - bulge mass)
    DiskMass = StellarMass - BulgeMass
    
    # Filter for galaxies with positive stellar mass and valid disk radius
    w = np.where((StellarMass > 0.0) & (DiskRadius > 0.0) & (Type == 0))[0]
    
    if len(w) > 0:
        # Calculate disk contribution to central density
        # For an exponential disk profile: Sigma(R) = Sigma_0 * exp(-R/R_d)
        R_d_kpc = DiskRadius[w] * 1000.0  # Convert from Mpc/h to kpc
        R_inner = 1.0  # kpc
        
        # Disk contribution (only if DiskMass > 0)
        disk_enclosed_mass = np.zeros(len(w))
        disk_mask = DiskMass[w] > 0.0
        if np.any(disk_mask):
            ratio_disk = R_inner / R_d_kpc[disk_mask]
            enclosed_mass_fraction_disk = 1.0 - np.exp(-ratio_disk) * (1.0 + ratio_disk)
            disk_enclosed_mass[disk_mask] = DiskMass[w][disk_mask] * enclosed_mass_fraction_disk
        
        # Bulge contribution to central density
        # Assuming a Hernquist or Sersic profile, bulge is very concentrated
        # For simplicity, assume bulge with scale radius R_b
        # Mass within radius r for Hernquist: M(<r) = M_bulge * r^2 / (r + R_b)^2
        bulge_enclosed_mass = np.zeros(len(w))
        bulge_mask = BulgeMass[w] > 0.0
        if np.any(bulge_mask):
            R_b_kpc = BulgeRadius[w][bulge_mask] * 1000.0  # Convert to kpc
            # For Hernquist profile
            bulge_enclosed_mass[bulge_mask] = BulgeMass[w][bulge_mask] * R_inner**2 / (R_inner + R_b_kpc)**2
        
        # Total enclosed mass within 1 kpc
        total_enclosed_mass = disk_enclosed_mass + bulge_enclosed_mass
        area_1kpc = np.pi * R_inner**2
        Sigma_1kpc = total_enclosed_mass / area_1kpc
        
        # Calculate sSFR for these galaxies
        sSFR_filtered = np.log10((SfrDisk[w] + SfrBulge[w]) / StellarMass[w])
        
        # Define green valley boundaries (typically -11 to -10.5 or similar)
        green_valley_upper = sSFRcut  # -11.0
        green_valley_lower = sSFRcut - 0.5  # -11.5
        
        # Separate into star forming, green valley, and quiescent
        sf_mask = sSFR_filtered > green_valley_upper
        gv_mask = (sSFR_filtered <= green_valley_upper) & (sSFR_filtered > green_valley_lower)
        q_mask = sSFR_filtered <= green_valley_lower
        
        # Sample galaxies for plotting
        if len(w) > dilute:
            indices_to_sample = sample(range(len(w)), dilute)
        else:
            indices_to_sample = range(len(w))
        
        # Get sampled indices for each population
        sf_indices = [i for i in indices_to_sample if sf_mask[i]]
        gv_indices = [i for i in indices_to_sample if gv_mask[i]]
        q_indices = [i for i in indices_to_sample if q_mask[i]]
        
        # Plot scatter - star forming (first, so it's on bottom)
        if len(sf_indices) > 0:
            plt.scatter(np.log10(StellarMass[w[sf_indices]]), np.log10(Sigma_1kpc[sf_indices]), 
                       c='dodgerblue', s=5, edgecolors='none', marker='s', alpha=0.6)
        
        # Plot scatter - green valley
        if len(gv_indices) > 0:
            plt.scatter(np.log10(StellarMass[w[gv_indices]]), np.log10(Sigma_1kpc[gv_indices]), 
                       c='mediumseagreen', s=5, edgecolors='none', marker='s')
        
        # Plot scatter - quiescent (last, so it's on top)
        if len(q_indices) > 0:
            plt.scatter(np.log10(StellarMass[w[q_indices]]), np.log10(Sigma_1kpc[q_indices]), 
                       c='firebrick', s=5, edgecolors='none', marker='s')

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} \Sigma_{<1\mathrm{kpc}}\ (M_{\odot}\ \mathrm{kpc}^{-2})$')
    plt.xlim(8.0, 12.0)
    plt.ylim(6.5, 10.5)
    
    # Create custom legend with colored text and no markers
    from matplotlib.patches import Rectangle
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels]
    legend = plt.legend(handles, legend_labels, loc='upper left', frameon=False)
    for i, text in enumerate(legend.get_texts()):
        text.set_color(legend_colors[i])
    
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '32.central_stellar_density' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting Kennicutt-Schmidt relation')

    plt.figure()
    ax = plt.subplot(111)

    # Filter for galaxies with positive cold gas and disk radius
    # Include total SFR (disk + bulge)
    w = np.where((ColdGas > 0.0) & (DiskRadius > 0.0))[0]
    
    print(f'Found {len(w)} galaxies with cold gas and disk radius')
    
    if len(w) > 0:
        # Calculate gas surface density
        # DiskRadius is in Mpc/h, convert to physical kpc
        R_d_kpc = DiskRadius[w] * 1000.0 / Hubble_h  # kpc (physical)
        
        # Match the model's disk area: π × (3 × R_d)² to be consistent
        # This captures ~95% of exponential disk mass
        disk_area = np.pi * (3*R_d_kpc)**2  # kpc^2 (physical)
        
        # Use total cold gas for KS relation
        Sigma_gas_kpc = ColdGas[w] / disk_area  # M_sun / kpc^2
        Sigma_gas = Sigma_gas_kpc / 1e6  # M_sun / pc^2
        
        # Calculate SFR surface density (use total SFR: disk + bulge)
        # Keep in kpc^-2
        total_SFR = SfrDisk[w] + SfrBulge[w]
        Sigma_SFR = total_SFR / disk_area  # M_sun/yr / kpc^2
        
        # Filter out galaxies with zero SFR for plotting
        w_sfr = np.where(Sigma_gas > -10.0)[0]
        print(f'Of those, {len(w_sfr)} have SFR > 0')
        print(f'Sigma_gas range: {np.min(Sigma_gas[w_sfr]):.2e} to {np.max(Sigma_gas[w_sfr]):.2e} M_sun/pc^2')
        print(f'Sigma_SFR range: {np.min(Sigma_SFR[w_sfr]):.2e} to {np.max(Sigma_SFR[w_sfr]):.2e} M_sun/yr/kpc^2')
        
        if len(w_sfr) == 0:
            print('No galaxies with positive SFR!')
        else:
            Sigma_gas_plot = Sigma_gas[w_sfr]
            Sigma_SFR_plot = Sigma_SFR[w_sfr]
            w_indices = w[w_sfr]
            w_indices = w[w_sfr]
        
            # Sample for plotting
            if len(w_sfr) > dilute:
                indices = sample(range(len(w_sfr)), dilute)
            else:
                indices = range(len(w_sfr))
            
            # Calculate sSFR for coloring
            sSFR_filtered = np.log10((SfrDisk[w_indices] + SfrBulge[w_indices]) / StellarMass[w_indices])
            green_valley_upper = sSFRcut
            green_valley_lower = sSFRcut - 0.5
            
            sf_mask = sSFR_filtered > green_valley_upper
            gv_mask = (sSFR_filtered <= green_valley_upper) & (sSFR_filtered > green_valley_lower)
            q_mask = sSFR_filtered <= green_valley_lower
            
            # Get sampled indices for each population
            sf_indices = [i for i in indices if sf_mask[i]]
            gv_indices = [i for i in indices if gv_mask[i]]
            q_indices = [i for i in indices if q_mask[i]]
            
            # Plot scatter - star forming
            if len(sf_indices) > 0:
                plt.scatter(np.log10(Sigma_gas_plot[sf_indices]), np.log10(Sigma_SFR_plot[sf_indices]),
                           c='dodgerblue', s=5, edgecolors='none', alpha=0.6)
            
            # Plot scatter - green valley
            if len(gv_indices) > 0:
                plt.scatter(np.log10(Sigma_gas_plot[gv_indices]), np.log10(Sigma_SFR_plot[gv_indices]),
                           c='mediumseagreen', s=5, edgecolors='none')
            
            # Plot scatter - quiescent
            if len(q_indices) > 0:
                plt.scatter(np.log10(Sigma_gas_plot[q_indices]), np.log10(Sigma_SFR_plot[q_indices]),
                           c='firebrick', s=5, edgecolors='none')
            
            # Plot the canonical KS relation: Sigma_SFR = A * Sigma_gas^N
            # Kennicutt (1998): Sigma_SFR [M_sun yr^-1 kpc^-2] = 2.5e-4 * (Sigma_gas [M_sun pc^-2])^1.4
            gas_range = np.logspace(-1, 4, 100)  # M_sun/pc^2
            kennicutt_sfr = 2.5e-4 * gas_range**1.4  # M_sun/yr/kpc^2
            plt.plot(np.log10(gas_range), np.log10(kennicutt_sfr), 'k--', linewidth=2, 
                    label='Kennicutt (1998), N=1.4', alpha=0.7)

    plt.xlabel(r'$\log_{10} \Sigma_{\mathrm{gas}}\ (M_{\odot}\ \mathrm{pc}^{-2})$')
    plt.ylabel(r'$\log_{10} \Sigma_{\mathrm{SFR}}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{kpc}^{-2})$')
    plt.xlim(-0.5, 5)
    plt.ylim(-4, 3)
    
    # Create custom legend
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent', 'Kennicutt (1998), N=1.4']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick', 'black']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels[:-1]]
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2))
    legend = plt.legend(handles, legend_labels, loc='upper left', frameon=False)
    for i, text in enumerate(legend.get_texts()[:-1]):
        text.set_color(legend_colors[i])
    
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '33.kennicutt_schmidt_relation' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting black hole mass vs stellar mass colored by SFR')

    plt.figure()
    ax = plt.subplot(111)

    # Filter for galaxies with positive black hole mass and stellar mass
    w = np.where((BlackHoleMass > 0.0) & (StellarMass > 0.0) & (Type == 0))[0]
    
    print(f'Found {len(w)} central galaxies with black holes')
    
    if len(w) > 0:
        # Calculate total SFR and sSFR
        total_SFR = SfrDisk[w] + SfrBulge[w]
        sSFR = total_SFR / StellarMass[w]  # yr^-1
        
        # Sample for plotting
        if len(w) > dilute:
            indices = sample(range(len(w)), dilute)
        else:
            indices = range(len(w))
        
        # Use log10(sSFR) for coloring, handle zero/negative values
        log_sSFR = np.log10(np.maximum(sSFR[indices], 1e-15))
        
        # Create scatter plot
        scatter = ax.scatter(np.log10(StellarMass[w[indices]]), np.log10(BlackHoleMass[w[indices]]),
                           c=log_sSFR, s=5, cmap='coolwarm_r', 
                           vmin=-13, vmax=-9, edgecolors='none', alpha=0.9, marker='s')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\log_{10}$ sSFR $(yr^{-1})$', fontsize=12)
        
        # Add observed M_BH - M_* relations
        stellar_mass_range = np.logspace(9, 12, 100)

    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{BH}}\ (M_{\odot})$')
    plt.xlim(9, 12)
    plt.ylim(5, 10)
    plt.legend(loc='upper left', frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '34.black_hole_mass_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting sSFR vs black hole mass / stellar mass')

    plt.figure()
    ax = plt.subplot(111)

    # Filter for galaxies with positive black hole mass, stellar mass, and SFR
    total_SFR_all = SfrDisk + SfrBulge
    w = np.where((BlackHoleMass > 0.0) & (StellarMass > 0.0) & (total_SFR_all > 0.0) & (Type == 0))[0]
    
    print(f'Found {len(w)} central galaxies with black holes and SFR > 0')
    
    if len(w) > 0:
        # Calculate sSFR and BH mass fraction
        sSFR = total_SFR_all[w] / StellarMass[w]
        BH_mass_fraction = BlackHoleMass[w] / StellarMass[w]
        
        # Sample for plotting
        if len(w) > dilute:
            indices = sample(range(len(w)), dilute)
        else:
            indices = range(len(w))
        
        # Calculate log values
        log_sSFR = np.log10(np.maximum(sSFR[indices], 1e-15))
        log_BH_fraction = np.log10(BH_mass_fraction[indices])
        
        # Separate into star forming, green valley, and quiescent
        sf_mask = log_sSFR > sSFRcut
        gv_mask = (log_sSFR <= sSFRcut) & (log_sSFR > sSFRcut - 0.5)
        q_mask = log_sSFR <= sSFRcut - 0.5
        
        # Get indices for each population
        sf_indices = [i for i in range(len(indices)) if sf_mask[i]]
        gv_indices = [i for i in range(len(indices)) if gv_mask[i]]
        q_indices = [i for i in range(len(indices)) if q_mask[i]]
        
        # Plot scatter - star forming
        if len(sf_indices) > 0:
            plt.scatter(log_BH_fraction[sf_indices], log_sSFR[sf_indices],
                       c='dodgerblue', s=5, edgecolors='none', alpha=0.9, marker='s')
        
        # Plot scatter - green valley
        if len(gv_indices) > 0:
            plt.scatter(log_BH_fraction[gv_indices], log_sSFR[gv_indices],
                       c='mediumseagreen', s=5, edgecolors='none', alpha=0.9, marker='s')
        
        # Plot scatter - quiescent
        if len(q_indices) > 0:
            plt.scatter(log_BH_fraction[q_indices], log_sSFR[q_indices],
                       c='firebrick', s=5, edgecolors='none', alpha=0.9, marker='s')

    plt.xlabel(r'$\log_{10} (M_{\mathrm{BH}} / M_{\mathrm{stars}})$')
    plt.ylabel(r'$\log_{10}$ sSFR $(yr^{-1})$')
    plt.xlim(-5.5, -1.5)
    plt.ylim(-14, -9)
    
    # Create custom legend with colored text and no markers
    from matplotlib.patches import Rectangle
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels]
    legend = plt.legend(handles, legend_labels, loc='lower left', frameon=False)
    for i, text in enumerate(legend.get_texts()):
        text.set_color(legend_colors[i])
    
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '35.ssfr_vs_bh_mass_fraction' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting black hole mass vs central stellar density within 1 kpc')

    plt.figure()
    ax = plt.subplot(111)

    # Calculate disk mass
    DiskMass = StellarMass - BulgeMass
    
    # Filter for galaxies with positive stellar mass, BH mass, and valid disk radius
    w = np.where((StellarMass > 0.0) & (DiskRadius > 0.0) & (BlackHoleMass > 0.0) & (Type == 0))[0]
    
    print(f'Found {len(w)} central galaxies with black holes for density calculation')
    
    if len(w) > 0:
        # Calculate disk contribution to central density
        R_d_kpc = DiskRadius[w] * 1000.0  # Convert from Mpc/h to kpc
        R_inner = 1.0  # kpc
        
        # Disk contribution (only if DiskMass > 0)
        disk_enclosed_mass = np.zeros(len(w))
        disk_mask = DiskMass[w] > 0.0
        if np.any(disk_mask):
            ratio_disk = R_inner / R_d_kpc[disk_mask]
            enclosed_mass_fraction_disk = 1.0 - np.exp(-ratio_disk) * (1.0 + ratio_disk)
            disk_enclosed_mass[disk_mask] = DiskMass[w][disk_mask] * enclosed_mass_fraction_disk
        
        # Bulge contribution using Hernquist profile
        bulge_enclosed_mass = np.zeros(len(w))
        bulge_mask = BulgeMass[w] > 0.0
        if np.any(bulge_mask):
            R_b_kpc = BulgeRadius[w][bulge_mask] * 1000.0  # Convert to kpc
            bulge_enclosed_mass[bulge_mask] = BulgeMass[w][bulge_mask] * R_inner**2 / (R_inner + R_b_kpc)**2
        
        # Total enclosed mass within 1 kpc
        total_enclosed_mass = disk_enclosed_mass + bulge_enclosed_mass
        area_1kpc = np.pi * R_inner**2
        Sigma_1kpc = total_enclosed_mass / area_1kpc
        
        # Calculate sSFR for coloring
        sSFR_filtered = np.log10((SfrDisk[w] + SfrBulge[w]) / StellarMass[w])
        
        # Define populations
        green_valley_upper = sSFRcut
        green_valley_lower = sSFRcut - 0.5
        
        sf_mask = sSFR_filtered > green_valley_upper
        gv_mask = (sSFR_filtered <= green_valley_upper) & (sSFR_filtered > green_valley_lower)
        q_mask = sSFR_filtered <= green_valley_lower
        
        # Sample for plotting
        if len(w) > dilute:
            indices = sample(range(len(w)), dilute)
        else:
            indices = range(len(w))
        
        # Get sampled indices for each population
        sf_indices = [i for i in indices if sf_mask[i]]
        gv_indices = [i for i in indices if gv_mask[i]]
        q_indices = [i for i in indices if q_mask[i]]
        
        # Plot scatter - star forming
        if len(sf_indices) > 0:
            plt.scatter(np.log10(Sigma_1kpc[sf_indices]), np.log10(BlackHoleMass[w[sf_indices]]),
                       c='dodgerblue', s=5, edgecolors='none', alpha=0.6)
        
        # Plot scatter - green valley
        if len(gv_indices) > 0:
            plt.scatter(np.log10(Sigma_1kpc[gv_indices]), np.log10(BlackHoleMass[w[gv_indices]]),
                       c='mediumseagreen', s=5, edgecolors='none')
        
        # Plot scatter - quiescent
        if len(q_indices) > 0:
            plt.scatter(np.log10(Sigma_1kpc[q_indices]), np.log10(BlackHoleMass[w[q_indices]]),
                       c='firebrick', s=5, edgecolors='none')

    plt.xlabel(r'$\log_{10} \Sigma_{<1\mathrm{kpc}}\ (M_{\odot}\ \mathrm{kpc}^{-2})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{BH}}\ (M_{\odot})$')
    plt.xlim(7, 11)
    plt.ylim(5, 10)
    
    # Create custom legend
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels]
    legend = plt.legend(handles, legend_labels, loc='upper left', frameon=False)
    for i, text in enumerate(legend.get_texts()):
        text.set_color(legend_colors[i])
    
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '36.bh_mass_vs_central_density' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting black hole mass vs virial velocity')

    plt.figure()
    ax = plt.subplot(111)

    # Filter for galaxies with positive black hole mass, stellar mass, and Vvir
    w = np.where((BlackHoleMass > 0.0) & (StellarMass > 0.0) & (Vvir > 0.0) & (Type == 0))[0]
    
    print(f'Found {len(w)} central galaxies with black holes and Vvir > 0')
    
    if len(w) > 0:
        # Calculate sSFR for coloring
        sSFR_filtered = np.log10((SfrDisk[w] + SfrBulge[w]) / StellarMass[w])
        
        # Define populations
        green_valley_upper = sSFRcut
        green_valley_lower = sSFRcut - 0.5
        
        sf_mask = sSFR_filtered > green_valley_upper
        gv_mask = (sSFR_filtered <= green_valley_upper) & (sSFR_filtered > green_valley_lower)
        q_mask = sSFR_filtered <= green_valley_lower
        
        # Sample for plotting
        if len(w) > dilute:
            indices = sample(range(len(w)), dilute)
        else:
            indices = range(len(w))
        
        # Get sampled indices for each population
        sf_indices = [i for i in indices if sf_mask[i]]
        gv_indices = [i for i in indices if gv_mask[i]]
        q_indices = [i for i in indices if q_mask[i]]
        
        # Plot scatter - star forming
        if len(sf_indices) > 0:
            plt.scatter(np.log10(Vvir[w[sf_indices]]), np.log10(BlackHoleMass[w[sf_indices]]),
                       c='dodgerblue', s=5, edgecolors='none', alpha=0.6)
        
        # Plot scatter - green valley
        if len(gv_indices) > 0:
            plt.scatter(np.log10(Vvir[w[gv_indices]]), np.log10(BlackHoleMass[w[gv_indices]]),
                       c='mediumseagreen', s=5, edgecolors='none')
        
        # Plot scatter - quiescent
        if len(q_indices) > 0:
            plt.scatter(np.log10(Vvir[w[q_indices]]), np.log10(BlackHoleMass[w[q_indices]]),
                       c='firebrick', s=5, edgecolors='none')

    plt.xlabel(r'$\log_{10} V_{\mathrm{vir}}\ (\mathrm{km}\ \mathrm{s}^{-1})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{BH}}\ (M_{\odot})$')
    plt.xlim(1.5, 3.0)
    plt.ylim(5, 10)
    
    # Create custom legend
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels]
    legend = plt.legend(handles, legend_labels, loc='upper left', frameon=False)
    for i, text in enumerate(legend.get_texts()):
        text.set_color(legend_colors[i])
    
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    outputFile = OutputDir + '37.bh_mass_vs_vvir' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting Mvir surface density around galaxies of different mass as a function of radii')

    plt.figure()
    ax = plt.subplot(111)

    # Define stellar mass bins for central galaxies (log10 M*)
    mass_bins = [
        (10.0, 10.4, r'$10.0 - 10.4$', 'navy'),
        (10.4, 10.7, r'$10.4 - 10.7$', 'blue'),
        (10.7, 11.0, r'$10.7 - 11.0$', 'cyan'),
        (11.0, 11.2, r'$11.0 - 11.2$', 'green'),
        (11.2, 11.4, r'$11.2 - 11.4$', 'yellow'),
        (11.4, 11.6, r'$11.4 - 11.6$', 'orange'),
        (11.6, 13.0, r'$> 11.6$', 'red')
    ]

    # Define radial bins in physical units (Mpc/h) - logarithmic spacing
    radial_bins = np.logspace(-2, np.log10(3), 15)  # 0.01 to 3 Mpc/h
    radial_centers = np.sqrt(radial_bins[:-1] * radial_bins[1:])  # geometric mean for log bins
    max_radius = radial_bins[-1]  # Maximum search radius

    # Get central galaxies only
    central_mask = (Type == 0) & (StellarMass > 0) & (Mvir > 0)
    centrals_idx = np.where(central_mask)[0]
    
    # Pre-filter galaxies with CentralMvir > 0 (all potential neighbors)
    neighbor_mask = CentralMvir > 0
    neighbor_idx = np.where(neighbor_mask)[0]
    neighbor_pos = np.column_stack([Posx[neighbor_mask], Posy[neighbor_mask], Posz[neighbor_mask]])
    neighbor_mvir = CentralMvir[neighbor_mask]
    neighbor_central_gal_idx = CentralGalaxyIndex[neighbor_mask]
    
    print(f'Total central galaxies: {len(centrals_idx)}')
    print(f'Total potential neighbors: {len(neighbor_idx)}')
    
    # For each mass bin, calculate surface density in each radial bin
    for mass_min, mass_max, label, color in mass_bins:
        # Select centrals in this mass range - mass bins are in log space
        log_mass = np.log10(StellarMass[centrals_idx])
        mass_mask = (log_mass >= mass_min) & (log_mass < mass_max)
        selected_centrals = centrals_idx[mass_mask]
        
        print(f'  Mass bin {label}: {len(selected_centrals)} central galaxies')
        
        if len(selected_centrals) == 0:
            continue
        
        # Initialize surface density profile for all radial bins
        surface_density_profile = []
        
        # Process all radial bins
        for r_idx in range(len(radial_bins) - 1):
            r_min = radial_bins[r_idx]  # Mpc/h
            r_max = radial_bins[r_idx + 1]  # Mpc/h
            
            total_mvir_in_bin = 0.0
            n_neighbors = 0
            unique_halos_seen = set()  # Track unique halos per bin
            
            # Vectorized processing: calculate distances for all centrals at once
            for central_idx in selected_centrals:
                # Get central position
                central_pos = np.array([Posx[central_idx], Posy[central_idx], Posz[central_idx]])
                
                # Exclude self if central is also in neighbor list
                not_self = neighbor_idx != central_idx
                
                # Calculate separation vectors with periodic boundary conditions
                dx = neighbor_pos[not_self, 0] - central_pos[0]
                dy = neighbor_pos[not_self, 1] - central_pos[1]
                dz = neighbor_pos[not_self, 2] - central_pos[2]
                
                # Apply periodic boundary conditions
                dx = np.where(dx > BoxSize/2, dx - BoxSize, dx)
                dx = np.where(dx < -BoxSize/2, dx + BoxSize, dx)
                dy = np.where(dy > BoxSize/2, dy - BoxSize, dy)
                dy = np.where(dy < -BoxSize/2, dy + BoxSize, dy)
                dz = np.where(dz > BoxSize/2, dz - BoxSize, dz)
                dz = np.where(dz < -BoxSize/2, dz + BoxSize, dz)
                
                # Calculate projected distance (2D in x-y plane)
                distance_projected = np.sqrt(dx**2 + dy**2)
                
                # Select galaxies in this radial bin
                in_bin = (distance_projected >= r_min) & (distance_projected < r_max)
                
                # Get the neighbor indices and halo IDs for galaxies in this bin
                neighbor_subset = neighbor_idx[not_self][in_bin]
                neighbor_central_gal_subset = neighbor_central_gal_idx[not_self][in_bin]
                neighbor_mvir_subset = neighbor_mvir[not_self][in_bin]
                
                # Count unique halos
                for idx, (gal_idx, halo_id, mvir_val) in enumerate(zip(neighbor_subset, 
                                                                         neighbor_central_gal_subset, 
                                                                         neighbor_mvir_subset)):
                    if halo_id not in unique_halos_seen:
                        unique_halos_seen.add(halo_id)
                        total_mvir_in_bin += mvir_val
                        n_neighbors += 1
            
            # Calculate surface density for this radial bin
            r_min_pc = r_min * 1e6 / Hubble_h  # pc (physical)
            r_max_pc = r_max * 1e6 / Hubble_h  # pc (physical)
            area_annulus = np.pi * (r_max_pc**2 - r_min_pc**2)  # pc^2
            
            total_mvir_h = total_mvir_in_bin * Hubble_h
            
            # Average surface density across all centrals
            if len(selected_centrals) > 0 and area_annulus > 0:
                surf_density = total_mvir_h / (area_annulus * len(selected_centrals))
                surface_density_profile.append(surf_density)
                print(f'    Bin {r_min:.3f}-{r_max:.3f} Mpc/h: {n_neighbors} unique halos, Σ={surf_density:.2e}')
            else:
                surface_density_profile.append(np.nan)
        
        # Plot the surface density profile
        surface_density_profile = np.array(surface_density_profile)
        
        valid = (~np.isnan(surface_density_profile)) & (surface_density_profile > 0)
        print(f'  Valid points: {np.sum(valid)} / {len(surface_density_profile)}')
        if np.sum(valid) > 0:
            plt.plot(radial_centers[valid], surface_density_profile[valid], 
                    color=color, label=label, linewidth=2, marker='o', markersize=5)

    plt.xlabel(r'$r_{\mathrm{p}}\ (\mathrm{Mpc}/h)$', fontsize=14)
    plt.ylabel(r'$\Sigma\ (h\,M_{\odot}\,\mathrm{pc}^{-2})$', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 10)
    plt.ylim(1e-2, 1e3)

    outputFile = OutputDir + '38.mvir_distribution_vs_radius' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting halo mass function')

    plt.figure()
    ax = plt.subplot(111)

    halos = (Mvir > 0)
    halo_masses = Mvir[halos]
    
    print(f'Found {len(halo_masses)} halos with Mvir > 0')

    # Create mass bins
    binwidth = 0.1  # dex
    log_masses = np.log10(halo_masses)
    mi = np.floor(log_masses.min()) - 1
    ma = np.floor(log_masses.max()) + 1
    NB = int((ma - mi) / binwidth)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(log_masses, range=(mi, ma), bins=NB)
    bin_centers = bin_edges[:-1] + 0.5 * binwidth
    
    # Convert to number density: divide by volume and bin width
    # phi = dn/dlog10M in units of (Mpc/h)^-3 dex^-1
    phi = counts / (volume * binwidth)
    
    # Plot
    w = np.where(phi > 0)[0]
    plt.plot(bin_centers[w], np.log10(phi[w]), 'b-', linewidth=2, label='SAGE26')
    
    plt.xlabel(r'$\log_{10}(M_{\mathrm{vir}}/M_{\odot})$')
    plt.ylabel(r'$\log_{10}[\phi\ \mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    plt.xlim(10, 15)

    outputFile = OutputDir + '39.halo_mass_function' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# --------------------------------------------------------

    print('Plotting SFR surface density vs SFR')

    plt.figure()
    ax = plt.subplot(111)

    # Calculate total SFR
    SFR = SfrDisk + SfrBulge
    
    # Filter for galaxies with positive SFR and disk radius
    w = np.where((SFR > 0) & (DiskRadius > 0) & (StellarMass > 0))[0]
    
    if len(w) > dilute:
        indices = sample(range(len(w)), dilute)
        w_sample = w[indices]
    else:
        w_sample = w
    
    print(f'Plotting {len(w_sample)} galaxies with SFR > 0')
    
    # Calculate SFR surface density
    # Σ_SFR = SFR / (π * R_disk^2)
    # DiskRadius is in Mpc/h, convert to kpc: R_kpc = DiskRadius * 1000 / h
    R_disk_kpc = DiskRadius[w_sample] * 1000.0 / Hubble_h  # kpc
    area_kpc2 = np.pi * R_disk_kpc**2  # kpc^2
    
    SFR_surface_density = SFR[w_sample] / area_kpc2  # M_sun/yr/kpc^2
    
    # Calculate sSFR for coloring
    sSFR_filtered = np.log10(SFR[w_sample] / StellarMass[w_sample])
    
    # Define populations
    green_valley_upper = sSFRcut
    green_valley_lower = sSFRcut - 0.5
    
    sf_mask = sSFR_filtered > green_valley_upper
    gv_mask = (sSFR_filtered <= green_valley_upper) & (sSFR_filtered > green_valley_lower)
    q_mask = sSFR_filtered <= green_valley_lower
    
    # Plot scatter - star forming
    if np.sum(sf_mask) > 0:
        plt.scatter(np.log10(SFR[w_sample[sf_mask]]), np.log10(SFR_surface_density[sf_mask]),
                   c='dodgerblue', s=5, edgecolors='none', alpha=0.6)
    
    # Plot scatter - green valley
    if np.sum(gv_mask) > 0:
        plt.scatter(np.log10(SFR[w_sample[gv_mask]]), np.log10(SFR_surface_density[gv_mask]),
                   c='mediumseagreen', s=5, edgecolors='none')
    
    # Plot scatter - quiescent
    if np.sum(q_mask) > 0:
        plt.scatter(np.log10(SFR[w_sample[q_mask]]), np.log10(SFR_surface_density[q_mask]),
                   c='firebrick', s=5, edgecolors='none')
    
    plt.xlabel(r'$\log_{10}\ \mathrm{SFR}\ (M_{\odot}\ \mathrm{yr}^{-1})$')
    plt.ylabel(r'$\log_{10}\ \Sigma_{\mathrm{SFR}}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{kpc}^{-2})$')
    
    # Create custom legend with colored text
    from matplotlib.patches import Rectangle
    legend_labels = ['Star Forming', 'Green Valley', 'Quiescent']
    legend_colors = ['dodgerblue', 'mediumseagreen', 'firebrick']
    handles = [Rectangle((0,0),1,1, fc="w", fill=False, edgecolor='none', linewidth=0) for _ in legend_labels]
    legend = plt.legend(handles, legend_labels, loc='upper left', frameon=False)
    for i, text in enumerate(legend.get_texts()):
        text.set_color(legend_colors[i])
    
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.xlim(-4, 3)
    plt.ylim(-6, 1)

    outputFile = OutputDir + '40.sfr_surface_density_vs_sfr' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting H1/M_star vs M_star')

    plt.figure()
    ax = plt.subplot(111)

    # HI mass now read directly from model output
    HI = H1gas
    HI[HI < 0] = 0  # Ensure non-negative

    # Select galaxies with stellar mass and HI
    w = np.where((StellarMass > 0) & (HI > 0) & (Type == 0))[0]
    
    if len(w) > dilute:
        w = sample(list(w), dilute)
    
    x = np.log10(StellarMass[w])
    y = np.log10(HI[w] / StellarMass[w])
    
    plt.scatter(x, y, c='cornflowerblue', s=1, alpha=0.3, rasterized=True)
    
    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.2)
    bin_centers = []
    bin_medians = []
    
    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))
    
    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

    data = read_obs_data('HIGasRatio_NonDetEQZero.dat')
    if data is not None:
        log_mstar = data[:, 0]
        median = data[:, 1]
        p16 = data[:, 2]
        p84 = data[:, 3]
        mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
        yerr_lower = np.abs(median[mask] - p16[mask])
        yerr_upper = np.abs(p84[mask] - median[mask])
        ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                    fmt='o', color='black', markersize=6, capsize=3, 
                    label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    else:
        print(f"  Warning: Could not add observational data: {e}")
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{HI}} / M_{\star})$')
    plt.xlim(8.0, 12.0)
    plt.ylim(-3, 2)

    outputFile = OutputDir + '41.HI_fraction_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting H2/M_star vs M_star')

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass and H2
    w = np.where((StellarMass > 0) & (H2gas > 0) & (Type == 0))[0]
    
    if len(w) > dilute:
        w = sample(list(w), dilute)
    
    x = np.log10(StellarMass[w])
    y = np.log10(H2gas[w] / StellarMass[w])
    
    plt.scatter(x, y, c='mediumorchid', s=1, alpha=0.3, rasterized=True)
    
    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.2)
    bin_centers = []
    bin_medians = []
    
    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))
    
    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

    try:
        data = read_obs_data('MolecularGasRatio_NonDetEQZero.dat')
        if data is not None:
            log_mstar = data[:, 0]
            median = data[:, 1]
            p16 = data[:, 2]
            p84 = data[:, 3]
            mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
            yerr_lower = np.abs(median[mask] - p16[mask])
            yerr_upper = np.abs(p84[mask] - median[mask])
            ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                       fmt='o', color='black', markersize=6, capsize=3, 
                       label='xCOLDGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    except Exception as e:
        print(f"  Warning: Could not add observational data: {e}")
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{H}_2} / M_{\star})$')
    plt.xlim(8.0, 12.0)
    plt.ylim(-3, 2)

    outputFile = OutputDir + '42.H2_fraction_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting H2/M_star vs M_star')

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass and H2
    w = np.where((StellarMass > 0) & (ColdGas > 0) & (Type == 0))[0]
    
    if len(w) > dilute:
        w = sample(list(w), dilute)
    
    x = np.log10(StellarMass[w])
    y = np.log10(ColdGas[w] / StellarMass[w])
    
    plt.scatter(x, y, c='deepskyblue', s=1, alpha=0.3, rasterized=True)
    
    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.2)
    bin_centers = []
    bin_medians = []
    
    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))
    
    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

    try:
        data = read_obs_data('NeutralGasRatio_NonDetEQZero.dat')
        if data is not None:
            log_mstar = data[:, 0]
            median = data[:, 1]
            p16 = data[:, 2]
            p84 = data[:, 3]
            mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
            yerr_lower = np.abs(median[mask] - p16[mask])
            yerr_upper = np.abs(p84[mask] - median[mask])
            ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                       fmt='o', color='black', markersize=6, capsize=3, 
                       label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    except Exception as e:
        print(f"  Warning: Could not add observational data: {e}")
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{cold}} / M_{\star})$')
    plt.xlim(8.0, 12.0)
    plt.ylim(-3, 2)

    outputFile = OutputDir + '45.Coldgas_fraction_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting H2 fraction (H2/ColdGas) vs M_star')

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass and cold gas
    w = np.where((StellarMass > 0) & (ColdGas > 0) & (Type == 0))[0]
    
    if len(w) > dilute:
        w = sample(list(w), dilute)
    
    x = np.log10(StellarMass[w])
    y = H2gas[w] / ColdGas[w]  # H2 fraction
    
    plt.scatter(x, y, c='forestgreen', s=1, alpha=0.3, rasterized=True)
    
    # Calculate median in bins with bootstrap errors
    mass_bins = np.arange(8.0, 12.5, 0.2)
    bin_centers = []
    bin_medians = []
    bin_lower = []
    bin_upper = []
    
    n_bootstrap = 1000
    
    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_data = y[mask]
            bin_medians.append(np.median(bin_data))
            
            # Bootstrap resampling
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                resample = np.random.choice(bin_data, size=len(bin_data), replace=True)
                bootstrap_medians.append(np.median(resample))
            
            # 68% confidence interval (1 sigma)
            bin_lower.append(np.percentile(bootstrap_medians, 16))
            bin_upper.append(np.percentile(bootstrap_medians, 84))
    
    if len(bin_centers) > 0:
        bin_centers = np.array(bin_centers)
        bin_medians = np.array(bin_medians)
        bin_lower = np.array(bin_lower)
        bin_upper = np.array(bin_upper)
        
        plt.fill_between(bin_centers, bin_lower, bin_upper, alpha=0.3, color='red', label='68% CI')
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')
        plt.legend(loc='best')
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$f_{\mathrm{H}_2}\ (M_{\mathrm{H}_2} / M_{\mathrm{cold}})$')
    plt.xlim(8.0, 12.0)
    plt.ylim(0, 1.0)

    outputFile = OutputDir + '43.H2_fraction_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting bulge fraction vs M_star')

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (Type == 0))[0]
    
    if len(w) > dilute:
        w = sample(list(w), dilute)
    
    x = np.log10(StellarMass[w])
    y = BulgeMass[w] / StellarMass[w]  # Bulge fraction
    
    plt.scatter(x, y, c='darkorange', s=1, alpha=0.3, rasterized=True)
    
    # Calculate median in bins with bootstrap errors
    mass_bins = np.arange(8.0, 12.5, 0.2)
    bin_centers = []
    bin_medians = []
    bin_lower = []
    bin_upper = []
    
    n_bootstrap = 1000
    
    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_data = y[mask]
            bin_medians.append(np.median(bin_data))
            
            # Bootstrap resampling
            bootstrap_medians = []
            for _ in range(n_bootstrap):
                resample = np.random.choice(bin_data, size=len(bin_data), replace=True)
                bootstrap_medians.append(np.median(resample))
            
            # 68% confidence interval (1 sigma)
            bin_lower.append(np.percentile(bootstrap_medians, 16))
            bin_upper.append(np.percentile(bootstrap_medians, 84))
    
    if len(bin_centers) > 0:
        bin_centers = np.array(bin_centers)
        bin_medians = np.array(bin_medians)
        bin_lower = np.array(bin_lower)
        bin_upper = np.array(bin_upper)
        
        plt.fill_between(bin_centers, bin_lower, bin_upper, alpha=0.3, color='red', label='68% CI')
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')
        plt.legend(loc='best')
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$M_{\mathrm{bulge}} / M_{\star}$')
    plt.xlim(8.0, 12.0)
    plt.ylim(0, 1.0)

    outputFile = OutputDir + '44.bulge_fraction_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (MetalsColdGas > 0))[0]

    if len(w) > dilute:
        w = sample(list(w), dilute)

    x = np.log10(StellarMass[w])
    y = np.log10(MetalsColdGas[w] / ColdGas[w])  # Gas metallicity

    plt.scatter(x, y, c='seagreen', s=1, alpha=0.3, rasterized=True)

    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.1)
    bin_centers = []
    bin_medians = []

    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))

    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ Z_{\mathrm{gas}}$')
    plt.xlim(8.0, 12.0)
    plt.ylim(-3, 0)

    outputFile = OutputDir + '46.gas_metallicity_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close() 

    # ---------------------------------------------------------

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (MetalsStellarMass > 0))[0]

    if len(w) > dilute:
        w = sample(list(w), dilute)

    x = np.log10(StellarMass[w])
    y = np.log10(MetalsStellarMass[w] / StellarMass[w])  # Stellar metallicity

    plt.scatter(x, y, c='purple', s=1, alpha=0.3, rasterized=True)

    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.1)
    bin_centers = []
    bin_medians = []

    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))

    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ Z_{\mathrm{stars}}$')
    plt.xlim(8.0, 12.0)
    plt.ylim(-3, 0)

    outputFile = OutputDir + '47.stellar_metallicity_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close() 

    # ------------------------------------------------------------

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (DiskRadius > 0))[0]
    DiskRadius = DiskRadius * 1.0e3 / Hubble_h
    DiskMass = StellarMass - BulgeMass

    if len(w) > dilute:
        w = sample(list(w), dilute)

    x = np.log10(DiskMass[w])
    y = np.log10(DiskRadius[w])

    plt.scatter(x, y, s=1, c='cornflowerblue', alpha=0.3, rasterized=True)

    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.1)
    bin_centers = []
    bin_medians = []

    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))

    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')


    plt.xlim(8,12)
    plt.ylim(-0.5, 2)

    plt.xlabel(r'$\log_{10}\ M_{*,\ disk}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ R_{*,\ disk}\ (kpc)$')

    outputFile = OutputDir + '48.disk_radius_vs_Mdisk' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # ------------------------------------------------------------

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (BulgeRadius > 0))[0]
    BulgeRadius = BulgeRadius * 1.0e3 / Hubble_h
    BulgeMass = BulgeMass

    if len(w) > dilute:
        w = sample(list(w), dilute)

    x = np.log10(BulgeMass[w])
    y = np.log10(BulgeRadius[w])

    plt.scatter(x, y, s=1, c='cornflowerblue', alpha=0.3, rasterized=True)

    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.1)
    bin_centers = []
    bin_medians = []

    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))

    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')

        
    plt.xlim(8,12)
    plt.ylim(-0.5, 2)

    plt.xlabel(r'$\log_{10}\ M_{*,\ bulge}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ R_{*,\ bulge}\ (kpc)$')

    outputFile = OutputDir + '49.bulge_radius_vs_Mbulge' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    # ------------------------------------------------------------

    plt.figure()
    ax = plt.subplot(111)

    # Select galaxies with stellar mass
    w = np.where((StellarMass > 0) & (DiskRadius > 0))[0]
    # DiskRadius = DiskRadius * 1.0e3 / Hubble_h

    if len(w) > dilute:
        w = sample(list(w), dilute)

    x = np.log10(StellarMass[w])
    y = np.log10(DiskRadius[w])

    plt.scatter(x, y, s=1, c='cornflowerblue', alpha=0.3, rasterized=True)

    # Calculate median in bins
    mass_bins = np.arange(8.0, 12.5, 0.1)
    bin_centers = []
    bin_medians = []

    for i in range(len(mass_bins)-1):
        mask = (x >= mass_bins[i]) & (x < mass_bins[i+1])
        if np.sum(mask) > 10:
            bin_centers.append((mass_bins[i] + mass_bins[i+1]) / 2.0)
            bin_medians.append(np.median(y[mask]))

    if len(bin_centers) > 0:
        plt.plot(bin_centers, bin_medians, 'r-', lw=2, label='Median')


    plt.xlim(8,12)
    plt.ylim(-0.5, 2)

    plt.xlabel(r'$\log_{10}\ M_{*}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ R_{*,\ disk}\ (kpc)$')

    outputFile = OutputDir + '50.disk_radius_vs_Mstar' + OutputFormat
    plt.savefig(outputFile)
    plt.close()

    print('\nAll plots completed!')
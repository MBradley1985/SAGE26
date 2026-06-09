#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
FileName = 'model_0.hdf5'

# Simulation details
Hubble_h = 0.73
BoxSize = 62.5
VolumeFraction = 1.0
FirstSnap = 0
LastSnap = 63
redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073,
             9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060,
             2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905,
             0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144,
             0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# Target redshifts and corresponding snapshots
target_redshifts = [0, 1, 2, 3, 4, 5, 6]
target_snaps = [63, 41, 32, 27, 24, 21, 18]

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['legend.facecolor'] = 'black'
plt.rcParams['legend.edgecolor'] = 'white'

# ==================================================================

def read_hdf(dirname, filename, snap_num, param):
    with h5.File(dirname + filename, 'r') as f:
        return np.array(f[snap_num][param])

def compute_smf(stellar_mass, volume, binwidth=0.1):
    """Compute stellar mass function from stellar masses."""
    w = np.where(stellar_mass > 0.0)[0]
    if len(w) == 0:
        return np.array([]), np.array([])
    mass = np.log10(stellar_mass[w])
    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth
    phi = counts / volume / binwidth
    return xaxeshisto, phi

def load_stellar_masses(dirname, snaps):
    """Load stellar masses for specified snapshots."""
    masses = {}
    for snap in snaps:
        snapshot = 'Snap_' + str(snap)
        masses[snap] = read_hdf(dirname, FileName, snapshot, 'StellarMass') * 1.0e10 / Hubble_h
    return masses

# ==================================================================

if __name__ == '__main__':

    print('Creating SMF comparison plots\n')

    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    OutputDir = './output/plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Color map for different redshifts
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(target_redshifts)))

    # ============================================================
    # Figure 1: noffb (solid) vs nocgm (dashed)
    # ============================================================

    print('Loading data for Figure 1: noffb vs nocgm')

    dir_noffb = './output/millennium_noffb/'
    dir_nocgm = './output/millennium_nocgm/'
    dir_vanilla = './output/millennium_vanilla/'

    masses_noffb = load_stellar_masses(dir_noffb, target_snaps)
    masses_nocgm = load_stellar_masses(dir_nocgm, target_snaps)
    masses_vanilla = load_stellar_masses(dir_vanilla, target_snaps)

    plt.figure()
    ax = plt.subplot(111)

    for i, (z, snap) in enumerate(zip(target_redshifts, target_snaps)):
        actual_z = redshifts[snap]

        # noffb - solid line
        x_noffb, phi_noffb = compute_smf(masses_noffb[snap], volume)
        if len(x_noffb) > 0:
            plt.plot(x_noffb, phi_noffb, '-', color=colors[i], lw=2,
                     label=f'z={z}' if i < 7 else None)

        # nocgm - dashed line
        x_nocgm, phi_nocgm = compute_smf(masses_nocgm[snap], volume)
        if len(x_nocgm) > 0:
            plt.plot(x_nocgm, phi_nocgm, '--', color=colors[i], lw=2)

        # vanilla - dotted line
        x_vanilla, phi_vanilla = compute_smf(masses_vanilla[snap], volume)
        if len(x_vanilla) > 0:
            plt.plot(x_vanilla, phi_vanilla, ':', color=colors[i], lw=2)

    plt.yscale('log')
    plt.axis([7.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}$)')
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.title('SMF: noffb (solid) vs nocgm (dashed) vs vanilla (dotted)')

    # Create custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'z={z}')
                       for i, z in enumerate(target_redshifts)]
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle='-', label='noffb'))
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle='--', label='nocgm'))
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle=':', label='vanilla'))

    leg = plt.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=9)
    leg.get_frame().set_alpha(0.8)

    plt.tight_layout()
    outputFile = OutputDir + 'SMF_noffb_vs_nocgm' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile)
    plt.close()

    # ============================================================
    # Figure 2: noffb (solid) vs c16feedback (dashed)
    # ============================================================

    print('Loading data for Figure 2: noffb vs c16feedback')

    dir_c16feedback = './output/millennium_c16feedback/'

    masses_c16feedback = load_stellar_masses(dir_c16feedback, target_snaps)

    plt.figure()
    ax = plt.subplot(111)

    for i, (z, snap) in enumerate(zip(target_redshifts, target_snaps)):
        actual_z = redshifts[snap]

        # noffb - solid line
        x_noffb, phi_noffb = compute_smf(masses_noffb[snap], volume)
        if len(x_noffb) > 0:
            plt.plot(x_noffb, phi_noffb, '-', color=colors[i], lw=2,
                     label=f'z={z}' if i < 7 else None)

        # c16feedback - dashed line
        x_c16, phi_c16 = compute_smf(masses_c16feedback[snap], volume)
        if len(x_c16) > 0:
            plt.plot(x_c16, phi_c16, '--', color=colors[i], lw=2)

        # vanilla - dotted line
        x_vanilla, phi_vanilla = compute_smf(masses_vanilla[snap], volume)
        if len(x_vanilla) > 0:
            plt.plot(x_vanilla, phi_vanilla, ':', color=colors[i], lw=2)

    plt.yscale('log')
    plt.axis([7.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}$)')
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.title('SMF: noffb (solid) vs c16feedback (dashed) vs vanilla (dotted)')

    # Create custom legend entries
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'z={z}')
                       for i, z in enumerate(target_redshifts)]
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle='-', label='noffb'))
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle='--', label='c16feedback'))
    legend_elements.append(Line2D([0], [0], color='white', lw=2, linestyle=':', label='vanilla'))

    leg = plt.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=9)
    leg.get_frame().set_alpha(0.8)

    plt.tight_layout()
    outputFile = OutputDir + 'SMF_noffb_vs_c16feedback' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile)
    plt.close()

    print('\nDone!')

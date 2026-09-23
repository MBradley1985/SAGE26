#!/usr/bin/env python
"""
Cold-gas number density of high-redshift SAGE26 galaxies, against the
Dekel et al. (2023) feedback-free-burst density threshold.

Dekel+23 (their Section 2) show that a starburst proceeds free of stellar-wind
and supernova feedback whenever the free-fall time is shorter than the ~1 Myr
delay before that feedback becomes effective (eq. 3), which -- via their
eq. (4) free-fall time / density relation -- is equivalent to a gas density

    n > n_fbk = 2.23e3 cm^-3                                        (eq. 5)

This script computes the mean cold-gas number density of each central galaxy,
n = ColdGas / [(4/3) pi R_disk^3] / (mu m_p) with mu = 1.2 (their adopted
mean molecular weight for neutral H+He gas at T < 10^4 K), and plots it
against stellar mass for a set of high-redshift snapshots, with n_fbk drawn
as a horizontal line. This is the density that FeedbackFreeModeOn=8
(model_regimes.c) effectively compares against once boosted by a clumping
factor -- FFBCloudClumping -- from the galaxy's mean disc density up to the
density of its actual star-forming clumps; this plot shows where the raw,
unclumped disc density itself already sits relative to n_fbk.

    python plotting/ffb_cold_gas_density.py
    python plotting/ffb_cold_gas_density.py --dir output/millennium_noffb --zmin 6 --zmax 15
    python plotting/ffb_cold_gas_density.py --min-mstar 0   # include every central, not just m* > 1e8

SAGE26 -- released under MIT (see LICENSE).
"""

import argparse, glob, os
import numpy as np
import h5py
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'kieren_cohare_palatino_sty.mplstyle')

MSUN_G = 1.989e33          # solar mass [g]
PROTON_MASS_G = 1.6726e-24 # proton mass [g]
MU_NEUTRAL = 1.2           # Dekel+23 sec. 2: mean molecular weight, neutral H+He, T < 10^4 K

# Dekel et al. (2023) eq. (5): feedback-free density threshold.
N_FBK_CM3 = 2.23e3

DEFAULT_DIR = './output/millennium_densityforffbregime/'
MIN_MSTAR = 1.0e8          # Msun; drop numerically noisy, near-empty galaxies

# One colour per redshift snapshot plotted, faintest (lowest z) to boldest.
Z_TARGETS = [6.2, 7.27, 8.55, 10.07, 11.9, 14.09]
CMAP = plt.get_cmap('viridis')


def model_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'No model_*.hdf5 in {directory}')
    return files


def header(directory):
    """Cosmology and unit conversions, from the run's own output."""
    with h5py.File(model_files(directory)[0], 'r') as f:
        sim, runtime = f['Header/Simulation'], f['Header/Runtime']
        h = float(sim.attrs['hubble_h'])
        unit_mass_g = float(runtime.attrs['UnitMass_in_g'])
        unit_length_cm = float(runtime.attrs['UnitLength_in_cm'])
        return {
            'hubble_h': h,
            'to_msun': unit_mass_g / MSUN_G / h,      # code mass -> Msun
            'to_cm': unit_length_cm / h,               # code length -> cm
            'z': np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps': list(f['Header/output_snapshots'][:]),
        }


def nearest_output_snap(hdr, z_target):
    """Output snapshot whose redshift is closest to z_target."""
    zs = hdr['z'][hdr['output_snaps']]
    i = int(np.argmin(np.abs(zs - z_target)))
    return hdr['output_snaps'][i], zs[i]


def read_snapshot(directory, hdr, snap, min_mstar):
    """Central galaxies' StellarMass, ColdGas and disc radius, in physical units.

    ColdGas and DiskRadius are converted here rather than left in code units:
    the density calculation needs grams and centimetres directly.
    """
    ms_list, cg_list, r_list = [], [], []
    for path in model_files(directory):
        with h5py.File(path, 'r') as f:
            grp = f.get(f'Snap_{snap}')
            if grp is None:
                continue
            # HDF5 stores these as float32; the physical (non-code-unit) masses
            # and cgs radii overflow float32's ~3.4e38 range, so promote first.
            ms = np.asarray(grp['StellarMass'], dtype=np.float64) * hdr['to_msun']
            keep = (np.asarray(grp['Type']) == 0) & (ms >= min_mstar)
            cg = np.asarray(grp['ColdGas'], dtype=np.float64) * hdr['to_msun']
            r = np.asarray(grp['DiskRadius'], dtype=np.float64) * hdr['to_cm']
            keep &= (cg > 0) & (r > 0)
            ms_list.append(ms[keep])
            cg_list.append(cg[keep])
            r_list.append(r[keep])
    return (np.concatenate(ms_list) if ms_list else np.array([]),
            np.concatenate(cg_list) if cg_list else np.array([]),
            np.concatenate(r_list) if r_list else np.array([]))


def cold_gas_density(coldgas_g_msun, r_cm):
    """Mean cold-gas number density within one disc scale radius [cm^-3].

    n = ColdGas / [(4/3) pi R^3] / (mu m_p), the same volume convention as
    Dekel+23 eq. (37) (mass within an effective radius, uniform sphere).
    """
    mass_g = coldgas_g_msun * MSUN_G
    volume_cm3 = (4.0 / 3.0) * np.pi * r_cm**3
    rho = mass_g / volume_cm3
    return rho / (MU_NEUTRAL * PROTON_MASS_G)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=DEFAULT_DIR, help=f'run directory (default: {DEFAULT_DIR})')
    ap.add_argument('--outdir', default=None, help='default: <dir>/plots/')
    ap.add_argument('--zmin', type=float, default=5.0, help='lowest redshift snapshot to include')
    ap.add_argument('--zmax', type=float, default=15.0, help='highest redshift snapshot to include')
    ap.add_argument('--min-mstar', type=float, default=MIN_MSTAR, help='Msun; 0 to disable')
    args = ap.parse_args()

    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    hdr = header(args.dir)
    z_targets = [z for z in Z_TARGETS if args.zmin <= z <= args.zmax]
    if not z_targets:
        raise SystemExit(f'No entries in Z_TARGETS fall within [{args.zmin}, {args.zmax}]')

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    colors = CMAP(np.linspace(0.15, 0.9, len(z_targets)))

    any_points = False
    for z_target, color in zip(z_targets, colors):
        snap, z_actual = nearest_output_snap(hdr, z_target)
        ms, cg, r = read_snapshot(args.dir, hdr, snap, args.min_mstar)
        if ms.size == 0:
            continue
        any_points = True
        n = cold_gas_density(cg, r)
        ax.scatter(ms, n, s=10, alpha=0.5, color=color, edgecolors='none',
                   label=rf'$z = {z_actual:.1f}$ ({ms.size} galaxies)', zorder=2)

    if not any_points:
        raise SystemExit('No galaxies passed the selection cuts in the requested redshift range.')

    ax.axhline(N_FBK_CM3, color='0.15', ls='--', lw=2.0, zorder=3)
    ax.text(0.02, N_FBK_CM3 * 1.2, r'Dekel+23 eq. 5: $n_{\rm fbk} = 2.23\times10^{3}\ \rm cm^{-3}$',
            transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=13)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, N_FBK_CM3 * 4.0))  # headroom for the threshold label
    ax.set_xlabel(r'Stellar Mass $(M_\odot)$')
    ax.set_ylabel(r'Cold-gas number density, $n\ (\mathrm{cm}^{-3})$')
    ax.legend(loc='lower right', frameon=False, fontsize=12, markerscale=2)

    outdir = args.outdir or os.path.join(args.dir, 'plots/')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'ColdGasDensity_FFBCriterion.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()

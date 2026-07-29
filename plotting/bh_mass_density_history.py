#!/usr/bin/env python3
"""
bh_mass_density_history.py
===========================
Cosmic (super)massive black hole mass density vs redshift:

    rho_BH(z) = sum(M_BH) / V_box   [M_sun Mpc^-3]

styled after the classic "BH mass density" figures in the SMBH-growth
literature (e.g. Shen 2009, ApJ, 704, 89; Marconi et al. 2004, MNRAS, 351,
169) -- an "Overall" curve (summed over all M_BH > 0) plus the separate
contributions from four black-hole mass decades:

    6 <= log(M_BH/Msun) <= 7    dotted
    7 <= log(M_BH/Msun) <= 8    densely dash-dot-dotted
    8 <= log(M_BH/Msun) <= 9    dash-dot
    9 <= log(M_BH/Msun) <= 10   dashed

Volume convention
------------------
Physical (little-h-free) volume, matching the BHMF convention in
allresults-blackholes.py / stellar-mass-density convention in
allresults-history.py:
    volume = (BoxSize / Hubble_h)^3 * VolumeFraction   [Mpc^3]
(NOT the (Mpc/h)^3 convention used for the accretion rate/luminosity
functions in bh_lrd_analysis.py -- there is no universal convention in the
literature for this particular plot, and the classic BH-mass-density papers
quote rho_BH in plain Msun/Mpc^3, so physical units are used here to match.)

z=0 anchor points (observational compilation)
----------------------------------------------
Shen (2009, ApJ, 704, 89):            rho_BH,0 ~ 4.0e5 Msun/Mpc^3,
                                       "uncertainty of a factor ~1.5"
Hopkins, Richards & Hernquist (2007,
  ApJ, 654, 731):                     rho_BH,0 = 4.81 (+1.24/-0.99) e5
Shankar, Salucci, Granato, De Zotti
  & Danese (2004, MNRAS, 354, 1020):  rho_BH,0 = 4.2 +/- 1.1 e5
Marconi, Risaliti, Gilli, Hunt,
  Maiolino & Salvati (2004, MNRAS,
  351, 169):                          rho_BH,0 = 4.6 (+1.9/-1.4) e5 h_0.7^2
                                       (~4.6e5 for h~0.7)
Graham & Driver (2007, MNRAS, 380,
  L15):                               rho_BH,0 ~ 4.4-5.9e5 (h=0.7 corrected)
Yu & Lu (2008, ApJ, 673, 1219):       rho_BH,0 ~ 4.0e5
  (uncertainty not independently confirmed for this script -- a
  representative +/-20% is assumed; treat that one error bar as
  illustrative, not a quoted literature value)

All values in Msun/Mpc^3, no h-dependence shown here since h~0.7 for all of
the above (the standard assumption in this literature when a paper's own h
scaling isn't explicitly carried through).

Usage
-----
    python3 plotting/bh_mass_density_history.py
    python3 plotting/bh_mass_density_history.py -i "output/millennium/model_*.hdf5"
    python3 plotting/bh_mass_density_history.py --zmax 10 --no-obs
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# MATPLOTLIB STYLE  (matches bh_lrd_analysis.py / allresults-blackholes.py)
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 140,
    'figure.autolayout': True,
    'font.family': 'serif',
    'font.size': 16.0,
    'axes.linewidth': 1.5,
    'xtick.major.size': 7.5, 'xtick.major.width': 1.5,
    'xtick.minor.size': 5.5, 'xtick.minor.width': 0.5,
    'xtick.direction': 'in', 'xtick.top': True, 'xtick.labelsize': 14,
    'ytick.major.size': 7.5, 'ytick.major.width': 1.5,
    'ytick.minor.size': 5.5, 'ytick.minor.width': 0.5,
    'ytick.direction': 'in', 'ytick.right': True, 'ytick.labelsize': 14,
    'legend.frameon': False, 'legend.fontsize': 11,
})

MILLENNIUM_BOX_MPC_H = 62.5   # comoving box side (Mpc/h) fallback

MASS_BINS = [(6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0)]
BIN_LINESTYLES = [':', (0, (3, 1, 1, 1)), '-.', '--']
BIN_LABELS = [
    r'$6 \leq \log M_{\rm BH}/M_\odot \leq 7$',
    r'$7 \leq \log M_{\rm BH}/M_\odot \leq 8$',
    r'$8 \leq \log M_{\rm BH}/M_\odot \leq 9$',
    r'$9 \leq \log M_{\rm BH}/M_\odot \leq 10$',
]

# z=0 observational compilation (rho_BH,0 in 1e5 Msun/Mpc^3, see docstring)
OBS_Z0 = [
    ('Shen+09',    4.0, 4.0 * (1 - 1/1.5), 4.0 * (1.5 - 1), '#4B0082'),
    ('Ho+07',      4.81, 0.99, 1.24, '#D32F2F'),
    ('Sh+04',      4.2, 1.1, 1.1, '#1976D2'),
    ('Ma+04',      4.6, 1.4, 1.9, '#388E3C'),
    ('Gr+07',      5.15, 0.75, 0.75, '#E040FB'),
    ('Yu+08',      4.0, 0.8, 0.8, '#FF9800'),
]

# ============================================================================
# I/O
# ============================================================================

def read_sim_params(filepath):
    """Read Hubble_h, BoxSize, VolumeFraction, snapshot_redshifts, available snaps."""
    out = {
        'hubble_h': 0.73, 'box_size': MILLENNIUM_BOX_MPC_H,
        'volume_fraction': 1.0, 'redshifts': None, 'available_snaps': [],
    }
    with h5py.File(filepath, 'r') as hf:
        if 'Header/Simulation' in hf:
            attrs = hf['Header/Simulation'].attrs
            out['hubble_h'] = float(attrs.get('hubble_h', out['hubble_h']))
            out['box_size'] = float(attrs.get('box_size', out['box_size']))
        if 'Header/Runtime' in hf and 'frac_volume_processed' in hf['Header/Runtime'].attrs:
            out['volume_fraction'] = float(hf['Header/Runtime'].attrs['frac_volume_processed'])
        if 'Header/snapshot_redshifts' in hf:
            out['redshifts'] = np.array(hf['Header/snapshot_redshifts'])
        out['available_snaps'] = sorted(
            int(k.split('_')[1]) for k in hf.keys() if k.startswith('Snap_'))
    return out


def read_black_hole_mass(file_list, snap_num, mass_conv):
    """Concatenate BlackHoleMass [Msun] across all input files at one snapshot."""
    out = []
    key = f'Snap_{snap_num}'
    for fpath in file_list:
        with h5py.File(fpath, 'r') as hf:
            if key in hf and 'BlackHoleMass' in hf[key]:
                arr = np.array(hf[key]['BlackHoleMass'])
                if arr.size:
                    out.append(arr)
    if not out:
        return np.array([])
    return np.concatenate(out) * mass_conv

# ============================================================================
# MAIN PLOT
# ============================================================================

def plot_bh_mass_density(file_list, output_file, zmax=None, show_obs=True):
    params = read_sim_params(file_list[0])
    hubble_h = params['hubble_h']
    box_size = params['box_size']
    redshifts = params['redshifts']
    snaps = params['available_snaps']

    total_volume_fraction = sum(read_sim_params(f)['volume_fraction'] for f in file_list)
    volume = (box_size / hubble_h)**3 * total_volume_fraction   # physical Mpc^3
    mass_conv = 1.0e10 / hubble_h

    print(f'Files:        {len(file_list)}')
    print(f'Hubble_h:     {hubble_h}')
    print(f'BoxSize:      {box_size} Mpc/h')
    print(f'VolumeFrac:   {total_volume_fraction}')
    print(f'Volume:       {volume:.4e} Mpc^3 (physical)')
    print(f'Snapshots:    {len(snaps)} available')

    z_arr = np.full(len(snaps), np.nan)
    rho_total = np.full(len(snaps), np.nan)
    rho_bins  = np.full((len(MASS_BINS), len(snaps)), np.nan)

    for i, sn in enumerate(snaps):
        if redshifts is not None and sn < len(redshifts):
            z_arr[i] = redshifts[sn]
        else:
            continue
        if zmax is not None and z_arr[i] > zmax:
            continue

        bh = read_black_hole_mass(file_list, sn, mass_conv)
        bh = bh[np.isfinite(bh) & (bh > 0)]
        if len(bh) == 0:
            rho_total[i] = 0.0
            continue

        rho_total[i] = bh.sum() / volume
        log_bh = np.log10(bh)
        for b, (lo, hi) in enumerate(MASS_BINS):
            m = (log_bh >= lo) & (log_bh < hi)
            rho_bins[b, i] = bh[m].sum() / volume if np.any(m) else 0.0

    ok = np.isfinite(z_arr) & np.isfinite(rho_total)
    z_arr, rho_total, rho_bins = z_arr[ok], rho_total[ok], rho_bins[:, ok]
    order = np.argsort(z_arr)
    z_arr, rho_total, rho_bins = z_arr[order], rho_total[order], rho_bins[:, order]

    print(f'z range plotted: {z_arr.min():.2f} - {z_arr.max():.2f}')
    print(f'log rho_BH(z=min): {np.log10(max(rho_total[0], 1e-30)):.3f}')

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.minorticks_on()

    x_hi = np.ceil(z_arr.max())

    if show_obs:
        ax.axvspan(-1.0, 0.0, color='#DDDDDD', alpha=0.6, zorder=0)

    nz = rho_total > 0
    ax.plot(z_arr[nz], np.log10(rho_total[nz]), color='k', lw=2.6, label='Overall (SAGE26)')

    for b, (ls, label) in enumerate(zip(BIN_LINESTYLES, BIN_LABELS)):
        y = rho_bins[b]
        m = y > 0
        if np.any(m):
            ax.plot(z_arr[m], np.log10(y[m]), color='k', lw=1.5, ls=ls, label=label)

    if show_obs:
        xpos = np.linspace(-0.9, -0.1, len(OBS_Z0))
        for (label, val, elo, ehi, color), x in zip(OBS_Z0, xpos):
            y = np.log10(val * 1.0e5)
            yerr_lo = y - np.log10((val - elo) * 1.0e5) if (val - elo) > 0 else 0.3
            yerr_hi = np.log10((val + ehi) * 1.0e5) - y
            ax.errorbar(x, y, yerr=[[yerr_lo], [yerr_hi]], fmt='o', ms=8,
                        color=color, mec='k', mew=0.6, capsize=3, zorder=5,
                        label=label)

    ax.set_xlim(-1.0 if show_obs else 0.0, x_hi)
    ax.set_ylim(0.0, 7.0)
    ax.set_xlabel(r'$z$', fontsize=18)
    ax.set_ylabel(r'$\log_{10}\,\rho_\bullet\ [M_\odot\,\mathrm{Mpc}^{-3}]$', fontsize=18)
    ax.set_title('BH Mass Density', fontsize=16)

    ax.legend(loc='upper right', fontsize=10.5, ncol=1, handlelength=2.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')

# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Cosmic black hole mass density vs redshift (SAGE26).')
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium/model_*.hdf5')
    p.add_argument('--zmax', type=float, default=8.0,
                   help='Maximum redshift to include (default 8, matching the '
                        'classic BH-mass-density figures; pass a larger value '
                        'or 0 for no cap to see the full high-z range).')
    p.add_argument('--no-obs', action='store_true',
                   help='Skip the z=0 observational compilation.')
    p.add_argument('--output', default=None)
    args = p.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        print(f'ERROR: no files matched "{args.input_pattern}"'); sys.exit(1)

    if args.output:
        out = Path(args.output)
    else:
        d = Path(files[0]).parent / 'plots'
        d.mkdir(exist_ok=True)
        out = d / 'bh_mass_density_history.png'

    zmax = None if (args.zmax is not None and args.zmax <= 0) else args.zmax
    plot_bh_mass_density(files, out, zmax=zmax, show_obs=(not args.no_obs))


if __name__ == '__main__':
    main()

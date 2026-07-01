#!/usr/bin/env python
"""
compare_stripping.py -- A/B comparison of SAGE26 runs for the satellite-stripping
scheme (PhysicalStrippingOn = 0 legacy vs 1 physical timescale).

Reads two (or more) SAGE26 output runs and overlays the diagnostics most
sensitive to satellite gas stripping, so the effect of the stripping toggle can
be read off directly:

  A. Satellite quenched fraction vs stellar mass   (sSFR < 10^-11 yr^-1)
  B. Satellite gas fraction vs stellar mass        ((Cold+Hot+CGM)/baryons)
  C. HI mass function split by type                (satellites vs centrals)
  D. Satellite baryon fraction vs Mvir             (Mbaryon / Mvir)

Each --run is LABEL:PATTERN, where PATTERN is a glob to the run's model HDF5
files, e.g.

    python compare_stripping.py \
        --run legacy:output/millennium_legacy/model_*.hdf5 \
        --run physical:output/millennium_physical/model_*.hdf5

Add a third --run to compare Option 1 as well.  Snapshot defaults to the latest
available (z~0); use -s to pick another.

SAGE26 -- released under MIT (see LICENSE).
"""

import argparse
import glob
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sSFRcut = -11.0          # log10(yr^-1): divides quiescent from star-forming
binwidth = 0.25          # dex, stellar-mass bins


# ---------------------------------------------------------------------------
# HDF5 helpers (kept in sync with allresults-local.py)
# ---------------------------------------------------------------------------
def read_simulation_params(filepath):
    """Read global simulation parameters from a model HDF5 header.

    Layout matches allresults-local.py: cosmology under Header/Simulation,
    processed-volume fraction under Header/Runtime.
    """
    params = {}
    with h5py.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['VolumeFraction'] = float(f['Header/Runtime'].attrs['frac_volume_processed'])
        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        snap_numbers = sorted(int(s.replace('Snap_', '')) for s in snap_groups)
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None
    return params


def get_snapshot_redshift(params, snap_num):
    zs = params['snapshot_redshifts']
    return zs[snap_num] if snap_num < len(zs) else None


def read_hdf(filepaths, snap_group, param):
    """Read and concatenate one property across a run's model files."""
    chunks = []
    for fp in filepaths:
        with h5py.File(fp, 'r') as f:
            if snap_group in f and param in f[snap_group]:
                chunks.append(np.array(f[snap_group][param]))
    if not chunks:
        return None
    return np.concatenate(chunks)


def load_run(pattern, snap_num_request):
    """Load the properties needed for the stripping diagnostics for one run."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  Error: no files match: {pattern}")
        sys.exit(1)

    p0 = read_simulation_params(files[0])
    h = p0['Hubble_h']

    # Total volume across all subfiles (for the mass function normalisation).
    vol_frac = 0.0
    for fp in files:
        vol_frac += read_simulation_params(fp)['VolumeFraction']
    volume = (p0['BoxSize'] / h) ** 3.0 * vol_frac    # (Mpc)^3

    snap_num = snap_num_request if snap_num_request is not None else p0['latest_snapshot']
    if snap_num not in p0['available_snapshots']:
        print(f"  Error: snapshot {snap_num} not in {pattern}")
        sys.exit(1)
    snap = f'Snap_{snap_num}'

    def mass(param):    # 10^10 Msun/h -> Msun
        arr = read_hdf(files, snap, param)
        return None if arr is None else arr * 1.0e10 / h

    run = {
        'volume': volume,
        'redshift': get_snapshot_redshift(p0, snap_num),
        'snap_num': snap_num,
        'Type': read_hdf(files, snap, 'Type'),
        'StellarMass': mass('StellarMass'),
        'ColdGas': mass('ColdGas'),
        'HotGas': mass('HotGas'),
        'CGMgas': mass('CGMgas'),
        'H1gas': mass('H1gas'),
        'BlackHoleMass': mass('BlackHoleMass'),
        'IntraClusterStars': mass('IntraClusterStars'),
        'EjectedMass': mass('EjectedMass'),
        'Mvir': mass('Mvir'),
        'SfrDisk': read_hdf(files, snap, 'SfrDisk'),
        'SfrBulge': read_hdf(files, snap, 'SfrBulge'),
    }
    return run


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------
def binned_fraction(logx, flag, lo, hi):
    """Fraction of objects with flag==True in each log-x bin."""
    edges = np.arange(lo, hi + binwidth, binwidth)
    centres, frac = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (logx >= a) & (logx < b)
        n = np.count_nonzero(m)
        if n >= 5:
            centres.append(0.5 * (a + b))
            frac.append(np.count_nonzero(flag[m]) / n)
    return np.array(centres), np.array(frac)


def binned_median(logx, y, lo, hi):
    edges = np.arange(lo, hi + binwidth, binwidth)
    centres, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (logx >= a) & (logx < b)
        if np.count_nonzero(m) >= 5:
            centres.append(0.5 * (a + b))
            med.append(np.median(y[m]))
    return np.array(centres), np.array(med)


def mass_function(logm, volume):
    lo, hi = 6.0, 12.5
    edges = np.arange(lo, hi + binwidth, binwidth)
    counts, _ = np.histogram(logm, bins=edges)
    centres = edges[:-1] + 0.5 * binwidth
    phi = counts / volume / binwidth      # dN / dlogM / Mpc^3
    return centres, phi


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def satellite_mask(run):
    """Satellites = Type >= 1 (subhalo satellites and orphans)."""
    return run['Type'] >= 1


def central_mask(run):
    return run['Type'] == 0


def ssfr_quenched(run):
    sm = run['StellarMass']
    sfr = run['SfrDisk'] + run['SfrBulge']
    with np.errstate(divide='ignore', invalid='ignore'):
        ssfr_lin = np.where(sm > 0, sfr / sm, 0.0)
    return ssfr_lin < 10.0 ** sSFRcut


def make_figure(runs, labels, colours, redshift, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axA, axB, axC, axD = axes.flatten()

    for run, lab, col in zip(runs, labels, colours):
        sat = satellite_mask(run)
        cen = central_mask(run)
        sm = run['StellarMass']
        good = sm > 0
        logsm = np.log10(np.where(good, sm, 1.0))

        # --- A. Satellite quenched fraction vs stellar mass ---
        q = ssfr_quenched(run)
        m = sat & good
        x, f = binned_fraction(logsm[m], q[m], 7.5, 12.0)
        axA.plot(x, f, '-o', color=col, label=lab, ms=4)

        # --- B. Satellite gas fraction vs stellar mass ---
        gas = run['ColdGas'] + run['HotGas'] + run['CGMgas']
        bary = gas + sm
        with np.errstate(divide='ignore', invalid='ignore'):
            fgas = np.where(bary > 0, gas / bary, np.nan)
        m = sat & good & np.isfinite(fgas)
        x, med = binned_median(logsm[m], fgas[m], 7.5, 12.0)
        axB.plot(x, med, '-o', color=col, label=lab, ms=4)

        # --- C. HI mass function, satellites (solid) vs centrals (dashed) ---
        hi = run['H1gas']
        if hi is not None:
            for mask_, ls, tag in ((sat, '-', 'sat'), (cen, '--', 'cen')):
                mm = mask_ & (hi > 0)
                if np.count_nonzero(mm) > 0:
                    xc, phi = mass_function(np.log10(hi[mm]), run['volume'])
                    nz = phi > 0
                    axC.plot(xc[nz], np.log10(phi[nz]), ls, color=col,
                             label=f'{lab} ({tag})')

        # --- D. Satellite baryon fraction vs Mvir ---
        mbary = (sm + gas + run['BlackHoleMass']
                 + run['IntraClusterStars'] + run['EjectedMass'])
        mvir = run['Mvir']
        with np.errstate(divide='ignore', invalid='ignore'):
            fbary = np.where(mvir > 0, mbary / mvir, np.nan)
        m = sat & (mvir > 0) & np.isfinite(fbary)
        x, med = binned_median(np.log10(mvir[m]), fbary[m], 9.5, 14.5)
        axD.plot(x, med, '-o', color=col, label=lab, ms=4)

    axA.set(xlabel=r'$\log_{10}(M_*\,/\,M_\odot)$', ylabel='Satellite quenched fraction',
            ylim=(0, 1.05))
    axA.set_title('A. Satellite quenched fraction')
    axB.set(xlabel=r'$\log_{10}(M_*\,/\,M_\odot)$',
            ylabel=r'$M_{\rm gas}/(M_{\rm gas}+M_*)$', ylim=(0, 1.05))
    axB.set_title('B. Satellite gas fraction (Cold+Hot+CGM)')
    axC.set(xlabel=r'$\log_{10}(M_{\rm HI}\,/\,M_\odot)$',
            ylabel=r'$\log_{10}(\phi\,/\,{\rm Mpc^{-3}\,dex^{-1}})$', ylim=(-6, 0))
    axC.set_title('C. HI mass function (solid=sat, dashed=cen)')
    axD.set(xlabel=r'$\log_{10}(M_{\rm vir}\,/\,M_\odot)$',
            ylabel=r'$M_{\rm baryon}/M_{\rm vir}$')
    axD.set_title('D. Satellite baryon fraction vs Mvir')

    for ax in (axA, axB, axC, axD):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ztxt = f'z = {redshift:.2f}' if redshift is not None else ''
    fig.suptitle(f'Satellite-stripping comparison   {ztxt}', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Wrote {out_path}')


# ---------------------------------------------------------------------------
def parse_run(spec):
    if ':' not in spec:
        print(f"Error: --run must be LABEL:PATTERN (got '{spec}')")
        sys.exit(1)
    label, pattern = spec.split(':', 1)
    return label, pattern


def main():
    ap = argparse.ArgumentParser(
        description='A/B comparison of SAGE26 satellite-stripping schemes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  %(prog)s --run legacy:output/legacy/model_*.hdf5 "
               "--run physical:output/physical/model_*.hdf5")
    ap.add_argument('--run', action='append', required=True, metavar='LABEL:PATTERN',
                    help='A run to compare, given as LABEL:PATTERN (repeatable).')
    ap.add_argument('-s', '--snapshot', type=int, default=None,
                    help='Snapshot number (default: latest available / z~0).')
    ap.add_argument('-o', '--output', type=str, default='stripping_comparison.png',
                    help='Output figure path (default: stripping_comparison.png).')
    args = ap.parse_args()

    if len(args.run) < 2:
        print('Error: give at least two --run entries to compare.')
        sys.exit(1)

    labels, runs = [], []
    for spec in args.run:
        label, pattern = parse_run(spec)
        print(f'Loading run "{label}" from {pattern}')
        run = load_run(pattern, args.snapshot)
        print(f'  snap {run["snap_num"]}  z={run["redshift"]:.3f}  '
              f'{np.count_nonzero(satellite_mask(run))} satellites')
        labels.append(label)
        runs.append(run)

    colours = plt.cm.tab10(np.linspace(0, 1, 10))[:len(runs)]
    make_figure(runs, labels, colours, runs[0]['redshift'], args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
energy_cap_test.py -- compare SAGE26 runs with and without an energy-conservation
bound on the FIRE supernova ejection term.

The FIRE branch computes

    E_FB = eps_halo * f_FIRE(V_vir, z) * 0.5 * m_* * eta_SN E_SN

so the effective SN energy coupling is eps_eff = eps_halo * f_FIRE, which is
unbounded and exceeds unity in dwarfs and at high redshift.  The bound replaces
eps_eff by min(eps_eff, C):

    C = infinity  -- current behaviour
    C = 2         -- E_FB <= the total SN energy available
    C = 1         -- E_FB <= half the SN energy available

This script plots the resulting observables side by side.

Usage:
    python plotting/energy_cap_test.py <label>=<dir> [<label>=<dir> ...] \
        --out output/energy_cap_test --tag millennium
"""

import argparse
import glob
import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

OBS_DIR = './data/'
SMF_BINS = np.arange(6.5, 12.5, 0.25)
MZR_BINS = np.arange(7.0, 12.0, 0.2)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def run_meta(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'no model files in {directory}')
    fvp = 0.0
    for fn in files:
        with h5.File(fn, 'r') as f:
            fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    with h5.File(files[0], 'r') as f:
        sim = dict(f['Header/Simulation'].attrs)
        zz = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k.split('_')[1]) for k in f.keys() if k.startswith('Snap_'))
    h = float(sim['hubble_h'])
    return dict(files=files, h=h, zz=zz, snaps=snaps,
                volume=(float(sim['box_size']) / h) ** 3 * fvp)


def load_snap(meta, snap, keys):
    out = {}
    for fn in meta['files']:
        with h5.File(fn, 'r') as f:
            g = f[f'Snap_{snap}']
            if g['StellarMass'].shape[0] == 0:
                continue
            for k in keys:
                out.setdefault(k, []).append(np.array(g[k]))
    if not out:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def nearest_snap(meta, z):
    valid = [s for s in meta['snaps'] if s < len(meta['zz'])]
    return min(valid, key=lambda s: abs(meta['zz'][s] - z))


# ----------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------

def obs_smf_z0():
    """Li & White (2009) z~0 SMF: columns are log_M, log_phi, -err, +err."""
    path = os.path.join(OBS_DIR, 'smf/SMF_Li2009.dat')
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], np.abs(d[:, 2]), r'Li \& White 2009'


def obs_smf_highz(zbin):
    """Return [(label, logM, log_phi)] for the requested integer redshift bin."""
    cfgs = [
        ('Stefanon+21', 'smf/stefanon_smf_2021.ecsv', 'phi', 1e-4, False),
        ('Navarro-Carrera+23', 'smf/navarro_carrera_smf_2023.ecsv', 'phi', 1e-4, False),
        ('Weibel+24', 'smf/weibel_smf_2024.ecsv', 'log_phi', 1.0, True),
    ]
    out = []
    for label, rel, phicol, scale, is_log in cfgs:
        path = os.path.join(OBS_DIR, rel)
        if not os.path.exists(path):
            continue
        rows = [ln.split() for ln in open(path)
                if ln.strip() and not ln.startswith('#')]
        hdr, body = rows[0], rows[1:]
        try:
            iz, im, ip = hdr.index('redshift_bin'), hdr.index('log_M'), hdr.index(phicol)
        except ValueError:
            continue
        m, p = [], []
        for r in body:
            if int(float(r[iz])) != zbin:
                continue
            m.append(float(r[im]))
            v = float(r[ip])
            p.append(v if is_log else np.log10(v * scale))
        if m:
            out.append((label, np.array(m), np.array(p)))
    return out


def obs_mzr():
    out = []
    for label, rel in [(r'Andrews \& Martini 2013', 'metallicity/MMAdrews13.dat'),
                       ('Curti+20', 'metallicity/Curti2020.dat')]:
        path = os.path.join(OBS_DIR, rel)
        if os.path.exists(path):
            d = np.loadtxt(path)
            out.append((label, d[:, 0], d[:, 1]))
    return out


def obs_sfrd():
    path = os.path.join(OBS_DIR, 'sfrd/MandD_sfrd_2014.ecsv')
    if not os.path.exists(path):
        return None
    z, psi = [], []
    for ln in open(path):
        if ln.startswith('#') or ln.startswith('Reference') or not ln.strip():
            continue
        p = ln.replace('"', ' ').split()
        try:
            zlo, zhi, _a, lpsi = float(p[-6]), float(p[-5]), float(p[-4]), float(p[-3])
        except (ValueError, IndexError):
            continue
        z.append(0.5 * (zlo + zhi))
        psi.append(lpsi)
    return np.array(z), np.array(psi)


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------

def smf(meta, snap):
    d = load_snap(meta, snap, ['StellarMass'])
    if d is None:
        return None, None
    sm = d['StellarMass'] * 1e10 / meta['h']
    sm = sm[sm > 0]
    n, _ = np.histogram(np.log10(sm), bins=SMF_BINS)
    phi = n / meta['volume'] / (SMF_BINS[1] - SMF_BINS[0])
    cen = 0.5 * (SMF_BINS[:-1] + SMF_BINS[1:])
    ok = phi > 0
    return cen[ok], np.log10(phi[ok])


def mzr(meta, snap):
    d = load_snap(meta, snap, ['StellarMass', 'ColdGas', 'MetalsColdGas'])
    sm = d['StellarMass'] * 1e10 / meta['h']
    cg = d['ColdGas']
    with np.errstate(divide='ignore', invalid='ignore'):
        fgas = cg / (d['StellarMass'] + cg)
    w = (sm > 1e7) & (fgas > 0.1) & (d['MetalsColdGas'] > 0) & (cg > 0)
    lm = np.log10(sm[w])
    Z = np.log10((d['MetalsColdGas'][w] / cg[w]) / 0.02) + 9.0
    cen, med = [], []
    for lo, hi in zip(MZR_BINS[:-1], MZR_BINS[1:]):
        m = (lm >= lo) & (lm < hi)
        if m.sum() >= 30:
            cen.append(0.5 * (lo + hi))
            med.append(np.median(Z[m]))
    return np.array(cen), np.array(med)


def histories(meta):
    """SFRD(z), stellar mass density(z), ejected mass density(z)."""
    z, sfrd, smd, ejd = [], [], [], []
    for s in sorted(meta['snaps']):
        if s >= len(meta['zz']):
            continue
        d = load_snap(meta, s, ['StellarMass', 'SfrDisk', 'SfrBulge', 'EjectedMass'])
        if d is None or d['StellarMass'].size < 20:
            continue
        z.append(meta['zz'][s])
        sfrd.append((d['SfrDisk'] + d['SfrBulge']).sum() / meta['volume'])
        smd.append(d['StellarMass'].sum() * 1e10 / meta['h'] / meta['volume'])
        ejd.append(d['EjectedMass'].sum() * 1e10 / meta['h'] / meta['volume'])
    o = np.argsort(z)
    return (np.array(z)[o], np.array(sfrd)[o], np.array(smd)[o], np.array(ejd)[o])


def gas_fraction(meta, snap):
    d = load_snap(meta, snap, ['StellarMass', 'ColdGas'])
    sm = d['StellarMass'] * 1e10 / meta['h']
    cg = d['ColdGas'] * 1e10 / meta['h']
    w = sm > 1e7
    lm = np.log10(sm[w])
    fg = cg[w] / (cg[w] + sm[w])
    cen, med = [], []
    for lo, hi in zip(MZR_BINS[:-1], MZR_BINS[1:]):
        m = (lm >= lo) & (lm < hi)
        if m.sum() >= 30:
            cen.append(0.5 * (lo + hi))
            med.append(np.median(fg[m]))
    return np.array(cen), np.array(med)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='label=directory')
    ap.add_argument('--out', default='output/energy_cap_test')
    ap.add_argument('--tag', default='run')
    args = ap.parse_args()

    stylesheet = './plotting/kieren_cohare_palatino_sty.mplstyle'
    if os.path.exists(stylesheet):
        plt.style.use(stylesheet)

    runs = []
    for spec in args.runs:
        label, _, directory = spec.partition('=')
        runs.append((label, run_meta(directory)))
    colours = ['#1b1b1b', '#0072B2', '#D55E00', '#009E73'][:len(runs)]
    styles = ['-', '--', '-.', ':'][:len(runs)]

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.0))

    # ---- SMF at four redshifts ----
    for ax, zt in zip(axes.flat[:4], [0.0, 2.0, 4.0, 6.0]):
        for (label, meta), c, ls in zip(runs, colours, styles):
            s = nearest_snap(meta, zt)
            x, y = smf(meta, s)
            if x is None:
                continue
            ax.plot(x, y, ls, color=c, lw=2.2,
                    label=f'{label} (z={meta["zz"][s]:.2f})')
        if zt == 0.0:
            o = obs_smf_z0()
            if o is not None:
                ax.errorbar(o[0], o[1], yerr=o[2], fmt='o', ms=4, color='0.45',
                            mfc='0.8', mec='0.3', lw=0.8, label=o[3], zorder=1)
        else:
            for oname, om, op in obs_smf_highz(int(round(zt))):
                ax.plot(om, op, 'o', ms=5, mfc='0.8', mec='0.3', ls='none',
                        label=oname, zorder=1)
        ax.set_xlim(6.8, 12.2)
        ax.set_ylim(-6.2, 0.2)
        ax.set_xlabel(r'$\log_{10}\ m_{*}\ [M_\odot]$')
        ax.set_ylabel(r'$\log_{10}\ \phi\ [{\rm Mpc^{-3}\,dex^{-1}}]$')
        ax.set_title(rf'SMF, $z \simeq {zt:.0f}$', loc='left')
        ax.legend(frameon=False, fontsize=9, loc='lower left')

    # ---- SFRD ----
    ax = axes.flat[4]
    for (label, meta), c, ls in zip(runs, colours, styles):
        z, sfrd, _, _ = histories(meta)
        ax.plot(z, np.log10(np.maximum(sfrd, 1e-8)), ls, color=c, lw=2.2, label=label)
    o = obs_sfrd()
    if o is not None:
        ax.plot(o[0], o[1], 'o', ms=4, mfc='0.8', mec='0.3', ls='none',
                label=r'Madau \& Dickinson 2014', zorder=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(-3.2, -0.4)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$\log_{10}\ \rho_{\rm SFR}\ [M_\odot\,{\rm yr^{-1}\,Mpc^{-3}}]$')
    ax.set_title('cosmic SFR density', loc='left')
    ax.legend(frameon=False, fontsize=9)

    # ---- stellar mass density ----
    ax = axes.flat[5]
    for (label, meta), c, ls in zip(runs, colours, styles):
        z, _, smd, _ = histories(meta)
        ax.plot(z, np.log10(np.maximum(smd, 1e-8)), ls, color=c, lw=2.2, label=label)
    ax.set_xlim(0, 10)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$\log_{10}\ \rho_{*}\ [M_\odot\,{\rm Mpc^{-3}}]$')
    ax.set_title('stellar mass density', loc='left')
    ax.legend(frameon=False, fontsize=9)

    # ---- ejected reservoir ----
    ax = axes.flat[6]
    for (label, meta), c, ls in zip(runs, colours, styles):
        z, _, _, ejd = histories(meta)
        ax.plot(z, np.log10(np.maximum(ejd, 1e-8)), ls, color=c, lw=2.2, label=label)
    ax.set_xlim(0, 10)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$\log_{10}\ \rho_{\rm ejected}\ [M_\odot\,{\rm Mpc^{-3}}]$')
    ax.set_title('ejected reservoir', loc='left')
    ax.legend(frameon=False, fontsize=9)

    # ---- MZR ----
    ax = axes.flat[7]
    for (label, meta), c, ls in zip(runs, colours, styles):
        x, y = mzr(meta, nearest_snap(meta, 0.0))
        ax.plot(x, y, ls, color=c, lw=2.2, label=label)
    for oname, om, oz in obs_mzr():
        ax.plot(om, oz, 'o', ms=5, mfc='0.8', mec='0.3', ls='none', label=oname, zorder=1)
    ax.set_xlim(7.0, 11.6)
    ax.set_ylim(7.3, 9.4)
    ax.set_xlabel(r'$\log_{10}\ m_{*}\ [M_\odot]$')
    ax.set_ylabel(r'$12 + \log_{10}({\rm O/H})$')
    ax.set_title('gas-phase MZR, $z=0$', loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower right')

    # ---- cold gas fraction ----
    ax = axes.flat[8]
    for (label, meta), c, ls in zip(runs, colours, styles):
        x, y = gas_fraction(meta, nearest_snap(meta, 0.0))
        ax.plot(x, y, ls, color=c, lw=2.2, label=label)
    ax.set_xlim(7.0, 11.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\log_{10}\ m_{*}\ [M_\odot]$')
    ax.set_ylabel(r'$M_{\rm cold} / (M_{\rm cold} + m_{*})$')
    ax.set_title('cold gas fraction, $z=0$', loc='left')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f'SN energy-conservation bound on $E_{{\\rm FB}}$ -- {args.tag}',
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ('.pdf', '.png'):
        p = os.path.join(args.out, f'energy_cap_{args.tag}{ext}')
        fig.savefig(p, dpi=110)
        print(f'  Saved: {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()

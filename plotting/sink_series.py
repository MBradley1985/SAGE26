#!/usr/bin/env python
"""
SAGE26 cooling-sink series
==========================
Companion to ``ablation_series.py``.  Where that script removes one *existing*
ingredient at a time, this one adds one *candidate fix* at a time for the
cooling-sink problem: SAGE26's only brake on the cooling flow is the AGN
``r_heat`` ratchet, which is gated on black-hole mass and therefore on bulge
mass, and so is weak at high redshift and strong at z = 0.  The model is
calibrated at z = 0, where the brake works, which hides the deficit at cosmic
noon.

Every variant differs from the *published* configuration by the switches listed
in ``VARIANTS`` and nothing else -- the baseline values in ``BASE_PINS`` are
written into every parameter file, so the series is reproducible whatever
``input/millennium.par`` currently holds.  Nothing is recalibrated: as in the
ablation series, the point is to isolate one change at fixed calibration.

The figure carries five diagnostics, each with a residual strip:

    stellar mass function at z = 0 and z = 2   (does the fix break the SMF?)
    cosmic SFR density vs redshift             (the cosmic-noon symptom)
    HI mass function at z = 0                  (the gas-hoarding symptom)
    H2 mass function at z = 0                  (does the fix move the molecular gas?)

Residuals are measured against the published baseline, so a flat line at zero
means "changed nothing".

Usage
-----
    python plotting/sink_series.py                  # plot from existing output
    python plotting/sink_series.py --run            # run any missing variants
    python plotting/sink_series.py --run --force    # re-run everything
    python plotting/sink_series.py --only base,fj055,ph6,rinc40

Must be run from the repository root (paths are relative, as in the .par files).
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_plots as pp


# ========================== CONFIGURATION ==========================

SAGE_BINARY = './sage'
BASE_PAR = 'input/millennium.par'
PAR_DIR = 'input'
OUT_ROOT = './output'

# Published configuration, forced into every variant so the baseline does not
# drift with whatever is currently in millennium.par.
BASE_PINS = {
    'DiskRadiusOn': '0',
    'DiskRadiusFactor': '1.0',
    'DiskRadiusMaxFrac': '0.15',
    'GasDiskRadiusFactor': '1.0',
    'PreventiveHeatingOn': '0',
    'PreventiveHeatingMass': '1.0e12',
    'PreventiveHeatingSlope': '2.0',
    'PreventiveHeatingEfficiency': '0.02',
    'ReIncorporationFactor': '0.15',
    'RadioModeEfficiency': '0.08',
    'SfrEfficiency': '0.05',
    'LastFile': '7',
}

REFERENCE_KEY = 'base'

VARIANTS = [
    {'key': 'base',   'label': r'published SAGE26', 'plain': 'published SAGE26',            'switch': {},
     'color': 'black', 'ls': '-',  'lw': 3.4, 'zorder': 12},

    # --- disc geometry: sets whether the disc goes unstable, hence whether a
    #     black hole grows at all, hence whether the AGN brake ever engages ---
    {'key': 'fj055',  'label': r'$f_j = 0.55$ (disc sizes)', 'plain': 'f_j = 0.55 (disc sizes)',
     'switch': {'DiskRadiusFactor': '0.55'},
     'color': '#0C5DA5', 'ls': (0, (6, 2)), 'lw': 2.3, 'zorder': 10},
    {'key': 'drs2',   'label': r'smoothed halo spin', 'plain': 'smoothed halo spin',
     'switch': {'DiskRadiusOn': '2'},
     'color': '#5599DD', 'ls': (0, (1, 1.4)), 'lw': 2.3, 'zorder': 9},

    # --- brakes on the cooling flow, three different gating variables ---
    {'key': 'ph3',    'label': r'brake: halo-mass gate', 'plain': 'brake: halo-mass gate',
     'switch': {'PreventiveHeatingOn': '3', 'PreventiveHeatingMass': '1.0e12'},
     'color': '#FF9500', 'ls': (0, (7, 2, 1.5, 2)), 'lw': 2.3, 'zorder': 10},
    {'key': 'ph5',    'label': r'brake: $t_{\rm cool}/t_{\rm ff}$ ceiling', 'plain': 'brake: t_cool/t_ff ceiling',
     'switch': {'PreventiveHeatingOn': '5'},
     'color': '#FF2C00', 'ls': (0, (3, 1.6)), 'lw': 2.3, 'zorder': 10},
    {'key': 'ph6',    'label': r'brake: accretion heating', 'plain': 'brake: accretion heating',
     'switch': {'PreventiveHeatingOn': '6', 'PreventiveHeatingEfficiency': '0.05'},
     'color': '#B5003C', 'ls': (0, (4, 1, 1, 1)), 'lw': 2.3, 'zorder': 10},

    # --- delay rather than removal ---
    {'key': 'rinc40', 'label': r'delayed reincorporation', 'plain': 'delayed reincorporation',
     'switch': {'ReIncorporationFactor': '0.40'},
     'color': '#00B945', 'ls': (0, (5, 1.5)), 'lw': 2.5, 'zorder': 11},

    # --- combinations ---
    {'key': 'ph6_rinc', 'label': r'accretion heating + delay', 'plain': 'accretion heating + delay',
     'switch': {'PreventiveHeatingOn': '6', 'PreventiveHeatingEfficiency': '0.05',
                'ReIncorporationFactor': '0.40'},
     'color': '#845B97', 'ls': '-', 'lw': 2.5, 'zorder': 8},
    {'key': 'fj_rinc',  'label': r'$f_j = 0.55$ + delay', 'plain': 'f_j = 0.55 + delay',
     'switch': {'DiskRadiusFactor': '0.55', 'ReIncorporationFactor': '0.40'},
     'color': '#00B0C0', 'ls': (0, (8, 2, 1, 2)), 'lw': 2.3, 'zorder': 8},

    # --- the brute-force comparison ---
    {'key': 'radio50',  'label': r'$\kappa_{\rm radio} = 0.5$', 'plain': 'radio efficiency = 0.5',
     'switch': {'RadioModeEfficiency': '0.5'},
     'color': '#777777', 'ls': (0, (10, 3)), 'lw': 2.1, 'zorder': 7},
]

# Panels ------------------------------------------------------------------
SMF_PANELS = [
    {'z': 0.0, 'tag': 'z=0', 'xlim': (8.0, 12.4), 'ylim': (-5.6, -0.8)},
    {'z': 2.0, 'tag': 'z=2', 'xlim': (8.0, 12.4), 'ylim': (-5.6, -0.8)},
]
BINWIDTH = 0.25
SMF_MASS_RANGE = (7.0, 13.0)
GAS_MASS_RANGE = (6.5, 11.5)
HIMF_XLIM, HIMF_YLIM = (7.5, 11.0), (-5.4, -0.6)
H2MF_XLIM, H2MF_YLIM = (7.0, 10.8), (-5.4, -0.8)
CSFRD_ZLIM, CSFRD_YLIM = (0.0, 8.0), (-3.2, -0.5)
RESIDUAL_YLIM = (-1.1, 1.1)
RESIDUAL_NEGLIGIBLE = 0.1
MIN_COUNT = 10          # bins below this many galaxies are drawn faint

OUTPUT_NAME = 'Sink_Series'
TABLE_SMF_MASSES = (10.0, 10.5, 11.0, 11.5)
TABLE_GAS_MASSES = (9.0, 9.5, 10.0, 10.5)
TABLE_REDSHIFTS = (0.0, 1.0, 2.0, 3.0, 4.0)


# ========================== PARAMETER FILES ==========================

def variant_par_path(key):
    return os.path.join(PAR_DIR, f'sink_{key}.par')


def variant_out_dir(key):
    return os.path.join(OUT_ROOT, f'sink_{key}') + os.sep


def write_variant_par(v):
    """
    Write the parameter file for *v*: the base file with BASE_PINS applied, then
    the variant's own switches on top, then OutputDir.

    A key absent from the base file is appended rather than silently dropped, so
    a newly added parameter does not fall back to its compiled-in default.
    """
    if not os.path.exists(BASE_PAR):
        sys.exit(f'{BASE_PAR} not found -- run from the repository root')
    settings = dict(BASE_PINS)
    settings.update(v['switch'])
    settings['OutputDir'] = variant_out_dir(v['key'])

    remaining = set(settings)
    lines = []
    with open(BASE_PAR) as fh:
        for line in fh:
            stripped = line.split('%', 1)[0].strip()
            name = stripped.split()[0] if stripped else None
            if name in settings:
                comment = line.split('%', 1)
                tail = ('   %' + comment[1].rstrip('\n')) if len(comment) > 1 else ''
                lines.append(f'{name:<27s} {settings[name]}{tail}\n')
                remaining.discard(name)
            else:
                lines.append(line)
    if remaining:
        lines.append('\n% ---- appended by plotting/sink_series.py ----\n')
        for name in sorted(remaining):
            lines.append(f'{name:<27s} {settings[name]}\n')

    path = variant_par_path(v['key'])
    with open(path, 'w') as fh:
        fh.writelines(lines)
    return path


def verify_par(v):
    """Re-read the written file and confirm it holds exactly the intended values."""
    path = variant_par_path(v['key'])
    got = {}
    with open(path) as fh:
        for line in fh:
            line = line.split('%', 1)[0].strip()
            if line:
                f = line.split()
                if len(f) >= 2:
                    got[f[0]] = f[1]
    want = dict(BASE_PINS)
    want.update(v['switch'])
    bad = [(k, want[k], got.get(k)) for k in want
           if got.get(k) is None or not _same_number(got[k], want[k])]
    return bad


def _same_number(a, b):
    try:
        return np.isclose(float(a), float(b), rtol=1e-12, atol=0.0)
    except ValueError:
        return a == b


def run_variants(variants, force=False):
    for v in variants:
        par = write_variant_par(v)
        bad = verify_par(v)
        if bad:
            for k, want, got in bad:
                print(f'    {v["key"]}: {k} = {got}, expected {want}')
            sys.exit(f'parameter file {par} does not hold the intended switches')
        out = variant_out_dir(v['key'])
        if pp.model_files_exist(out) and not force:
            print(f'  {v["key"]:>10s}: output present -- skipped')
            continue
        os.makedirs(out, exist_ok=True)
        print(f'  {v["key"]:>10s}: {SAGE_BINARY} {par}')
        res = subprocess.run([SAGE_BINARY, par],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            sys.stderr.write(res.stderr.decode(errors='replace')[-2000:])
            sys.exit(f'  {v["key"]}: SAGE failed (exit {res.returncode})')
    print()


# ========================== MEASUREMENT ==========================

def read_sim(directory):
    hdr = pp._read_sim_header(directory)
    if hdr is None:
        return None
    return {
        'box_size': hdr['box_size'],
        'hubble_h': hdr['hubble_h'],
        'volume': (hdr['box_size'] / hdr['hubble_h'])**3 * hdr['volume_fraction'],
        'volume_fraction': hdr['volume_fraction'],
        'mass_convert': hdr['unit_mass_in_g'] / pp._MSUN_CGS / hdr['hubble_h'],
        'redshifts': np.asarray(hdr['redshifts'], dtype=float),
    }


def _mf(path, snap, field, sim, mass_range):
    """Mass function of *field* at *snap*, plus the per-bin galaxy counts."""
    data = pp.read_snap_from_files(pp.find_model_files(path), f'Snap_{snap}',
                                   [field], mass_convert=sim['mass_convert'])
    if not data:
        return None, None, None
    m = data[field]
    m = m[m > 0]
    if m.size == 0:
        return None, None, None
    x, phi, _ = pp.mass_function(np.log10(m), sim['volume'],
                                 binwidth=BINWIDTH, mass_range=mass_range)
    edges = np.arange(mass_range[0], mass_range[1] + 0.5 * BINWIDTH, BINWIDTH)
    counts, _ = np.histogram(np.log10(m), bins=edges)
    return x, phi, counts[:x.size]


def measure(variants, sim):
    """Every diagnostic for every variant, on the shared bins of the reference."""
    results = {}
    for v in variants:
        path = variant_out_dir(v['key'])
        r = {}
        for panel in SMF_PANELS:
            snap = pp._snap_nearest_z(sim['redshifts'], panel['z'])
            x, phi, n = _mf(path, snap, 'StellarMass', sim, SMF_MASS_RANGE)
            r[f'smf_{panel["tag"]}'] = {'x': x, 'phi': phi, 'n': n,
                                        'z': sim['redshifts'][snap]}
        snap0 = pp._snap_nearest_z(sim['redshifts'], 0.0)
        for field, tag, rng in (('H1gas', 'himf', GAS_MASS_RANGE),
                                ('H2gas', 'h2mf', GAS_MASS_RANGE)):
            x, phi, n = _mf(path, snap0, field, sim, rng)
            r[tag] = {'x': x, 'phi': phi, 'n': n, 'z': sim['redshifts'][snap0]}

        z = sim['redshifts']
        rho = np.full(z.size, np.nan)
        files = pp.find_model_files(path)
        for snap in range(z.size):
            d = pp.read_snap_from_files(files, f'Snap_{snap}',
                                        ['SfrDisk', 'SfrBulge'],
                                        mass_convert=sim['mass_convert'])
            if not d:
                continue
            tot = float(np.sum(d['SfrDisk'] + d['SfrBulge']))
            if tot > 0:
                rho[snap] = tot / sim['volume']
        with np.errstate(divide='ignore', invalid='ignore'):
            r['csfrd'] = {'z': z, 'phi': np.log10(rho)}
        results[v['key']] = r
        print(f'  {v["key"]:>10s}: measured')
    print()
    return results


# ========================== OBSERVATIONS ==========================

def load_wright18(hubble_h):
    """
    Wright+18 SMF, grouped by median redshift.

    The file carries two independent determinations per (z, mass) -- they differ
    by up to 0.6 dex at the massive end -- so both are kept and drawn, rather
    than interpolated over, which would silently pick one at random.
    """
    path = './data/smf/Wright18_CombinedSMF.dat'
    if not os.path.exists(path):
        return {}
    h_w = 0.7
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if line.startswith('#') or len(p) < 6:
                continue
            try:
                mz, sm, ly = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                continue
            if ly < -10 or not np.isfinite(ly):
                continue
            phi = np.log10(10**(ly + np.log10(1 / 0.25)) * (h_w / hubble_h)**3)
            out[mz][round(sm, 3)].append(phi)
    return {z: {m: (min(v), max(v)) for m, v in d.items()} for z, d in out.items()}


def load_cosmos_web_csfrd():
    """COSMOS-Web CSFRD inferred from the stellar mass density (Chabrier)."""
    path = './data/sfrd/CSFRD_inferred_from_SMD.ecsv'
    if not os.path.exists(path):
        return None
    try:
        from astropy.table import Table
        t = Table.read(path, format='ascii.ecsv')
        return (np.asarray(t['Redshift'], float),
                np.log10(np.asarray(t['sfrd_50'], float)),
                np.log10(np.asarray(t['sfrd_16'], float)),
                np.log10(np.asarray(t['sfrd_84'], float)))
    except Exception as exc:
        print(f'  Warning: could not load COSMOS-Web CSFRD: {exc}')
        return None


def madau_dickinson(zz, imf='chabrier'):
    """
    Madau & Dickinson (2014) eq. 15.

    Their compilation and fit are quoted for a Salpeter IMF.  SAGE26's
    RecycleFraction of 0.43 is a Chabrier-like return fraction, so the fit is
    shifted by -0.24 dex to Chabrier before being compared -- the same footing
    as the COSMOS-Web curve, which is Chabrier natively.  Plotting one of the
    two curves in each IMF is a 0.24 dex inconsistency between observational
    datasets on the same axes.
    """
    psi = 0.015 * (1 + zz)**2.7 / (1 + ((1 + zz) / 2.9)**5.6)
    shift = pp.SALPETER_TO_CHABRIER_DEX if imf == 'chabrier' else 0.0
    return np.log10(psi) + shift


# ========================== FIGURE ==========================

def _guides(ax):
    ax.axhline(0.0, color='k', lw=0.9, alpha=0.6, zorder=1)
    ax.axhspan(-RESIDUAL_NEGLIGIBLE, RESIDUAL_NEGLIGIBLE,
               color='0.85', alpha=0.55, zorder=0)


def _fmt(ax, xmaj, xmin, ymaj, ymin, hide_x=False):
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(MultipleLocator(xmin))
    ax.yaxis.set_major_locator(MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(MultipleLocator(ymin))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if hide_x:
        ax.tick_params(labelbottom=False)


def _plot_mf(ax, axr, variants, results, key, ref, floor, xlim, ylim, xlabel):
    """One mass-function panel plus its residual strip."""
    for v in variants:
        m = results[v['key']][key]
        if m['x'] is None:
            continue
        good = np.isfinite(m['phi'])
        ax.plot(m['x'][good], m['phi'][good], color=v['color'], ls=v['ls'],
                lw=v['lw'], zorder=v['zorder'], label=v['label'])
        if v['key'] == REFERENCE_KEY:
            continue
        rm = ref[key]
        if rm['x'] is None:
            continue
        with np.errstate(invalid='ignore'):
            delta = m['phi'] - rm['phi']
        solid = good & np.isfinite(rm['phi']) & (rm['phi'] > floor)
        axr.plot(m['x'][solid], delta[solid], color=v['color'], ls=v['ls'],
                 lw=v['lw'], zorder=v['zorder'])
    ax.axhline(floor, color='0.6', lw=0.8, ls=':', zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _guides(axr)
    axr.set_ylim(*RESIDUAL_YLIM)
    axr.set_xlim(*xlim)
    axr.set_xlabel(xlabel)
    _fmt(ax, 1.0, 0.2, 1.0, 0.2, hide_x=True)
    _fmt(axr, 1.0, 0.2, 0.5, 0.1)


def make_figure(variants, results, sim, outdir):
    ref = results[REFERENCE_KEY]
    floor = float(np.log10(MIN_COUNT / sim['volume'] / BINWIDTH))
    wright = load_wright18(sim['hubble_h'])
    cw = load_cosmos_web_csfrd()

    fig = plt.figure(figsize=(17.5, 10.6))
    gs = GridSpec(4, 3, height_ratios=[3.0, 1.15, 3.0, 1.15],
                  hspace=0.06, wspace=0.22,
                  left=0.055, right=0.985, top=0.965, bottom=0.062)

    # ---- SMF panels ----
    for col, panel in enumerate(SMF_PANELS):
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)
        tag = f'smf_{panel["tag"]}'
        _plot_mf(ax, axr, variants, results, tag, ref, floor,
                 panel['xlim'], panel['ylim'],
                 r'$\log_{10}\ m_{*}\ [M_{\odot}]$')
        # observations: both determinations per bin, drawn as a vertical range
        zsel = min(wright, key=lambda q: abs(q - panel['z'])) if wright else None
        if zsel is not None and abs(zsel - panel['z']) < 0.75:
            for m, (lo, hi) in sorted(wright[zsel].items()):
                ax.plot([m, m], [lo, hi], color='0.35', lw=2.4, alpha=0.6,
                        solid_capstyle='butt', zorder=2)
            ax.plot([], [], color='0.35', lw=2.4, alpha=0.6,
                    label=rf'Wright+18 ($z={zsel:g}$, two estimates)')
        ax.text(0.95, 0.94, rf"$z = {ref[tag]['z']:.2f}$", transform=ax.transAxes,
                ha='right', va='top')
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
            axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')

    # ---- CSFRD ----
    ax = fig.add_subplot(gs[0, 2])
    axr = fig.add_subplot(gs[1, 2], sharex=ax)
    zz = np.linspace(*CSFRD_ZLIM, 300)
    ax.plot(zz, madau_dickinson(zz), color='0.55', lw=1.6, alpha=0.9, zorder=2,
            label=pp._tex_safe(r'Madau \& Dickinson 2014, shifted to Chabrier'))
    if cw is not None:
        zc, p50, p16, p84 = cw
        ax.fill_between(zc, p16, p84, color='#3366AA', alpha=0.22, zorder=2)
        ax.plot(zc, p50, color='#3366AA', lw=2.0, alpha=0.95, zorder=3,
                label='COSMOS-Web (from SMD)')
    for v in variants:
        r = results[v['key']]['csfrd']
        good = np.isfinite(r['phi'])
        ax.plot(r['z'][good], r['phi'][good], color=v['color'], ls=v['ls'],
                lw=v['lw'], zorder=v['zorder'])
        if v['key'] == REFERENCE_KEY or cw is None:
            continue
        with np.errstate(invalid='ignore'):
            delta = r['phi'] - np.interp(r['z'], zc, p50)
        axr.plot(r['z'][good], delta[good], color=v['color'], ls=v['ls'],
                 lw=v['lw'], zorder=v['zorder'])
    if cw is not None:
        rb = results[REFERENCE_KEY]['csfrd']
        gb = np.isfinite(rb['phi'])
        axr.plot(rb['z'][gb], rb['phi'][gb] - np.interp(rb['z'][gb], zc, p50),
                 color='black', ls='-', lw=3.4, zorder=12)
    ax.set_xlim(*CSFRD_ZLIM)
    ax.set_ylim(*CSFRD_YLIM)
    ax.set_ylabel(r'$\log_{10}\ \rho_{\rm SFR}\ [M_{\odot}\,{\rm yr}^{-1}\,{\rm Mpc}^{-3}]$')
    _guides(axr)
    axr.set_ylim(*RESIDUAL_YLIM)
    axr.set_xlabel(r'redshift')
    axr.set_ylabel(r'$\Delta$ vs COSMOS-Web')
    _fmt(ax, 2.0, 0.5, 0.5, 0.1, hide_x=True)
    _fmt(axr, 2.0, 0.5, 0.5, 0.1)
    ax.legend(loc='lower left', fontsize=9, frameon=False, ncol=1)

    # ---- HI mass function ----
    ax = fig.add_subplot(gs[2, 0])
    axr = fig.add_subplot(gs[3, 0], sharex=ax)
    _plot_mf(ax, axr, variants, results, 'himf', ref, floor,
             HIMF_XLIM, HIMF_YLIM, r'$\log_{10}\ M_{\rm HI}\ [M_{\odot}]$')
    obs_handles, obs_labels = [], []
    for od in pp.load_himf_observations():
        hh = ax.errorbar(od['mass'], od['phi'], fmt=od.get('marker', 'o'),
                         color='0.35', ms=5.5, markerfacecolor='0.75',
                         markeredgecolor='k', markeredgewidth=0.7, lw=0.9,
                         alpha=0.75, zorder=2)
        obs_handles.append(hh)
        obs_labels.append(od['label'])
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')
    if obs_handles:
        ax.legend(obs_handles, obs_labels, loc='lower left', fontsize=9, frameon=False)
    ax.text(0.95, 0.94, r'HI, $z = 0$', transform=ax.transAxes,
            ha='right', va='top')

    # ---- H2 mass function ----
    ax = fig.add_subplot(gs[2, 1])
    axr = fig.add_subplot(gs[3, 1], sharex=ax)
    _plot_mf(ax, axr, variants, results, 'h2mf', ref, floor,
             H2MF_XLIM, H2MF_YLIM, r'$\log_{10}\ M_{\rm H_2}\ [M_{\odot}]$')
    obs_handles, obs_labels = [], []
    for od in pp.load_h2mf_observations():
        hh = ax.errorbar(od['mass'], od['phi'], fmt=od.get('marker', 's'),
                         color='0.35', ms=5.5, markerfacecolor='0.75',
                         markeredgecolor='k', markeredgewidth=0.7, lw=0.9,
                         alpha=0.75, zorder=2)
        obs_handles.append(hh)
        obs_labels.append(od['label'])
    axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')
    if obs_handles:
        ax.legend(obs_handles, obs_labels, loc='lower left', fontsize=9, frameon=False)
    ax.text(0.95, 0.94, r'H$_2$, $z = 0$', transform=ax.transAxes,
            ha='right', va='top')

    # ---- legend panel ----
    axl = fig.add_subplot(gs[2:, 2])
    axl.axis('off')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    axl.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 0.90),
               fontsize=11.5, frameon=False, handlelength=3.4, borderaxespad=0.0,
               title='cooling-sink variants', title_fontsize=12.5)
    axl.text(0.0, 0.06, 'residuals are measured against the published model\n'
                        '(CSFRD residual against COSMOS-Web)\n'
                        f'dotted line / faint residuals: fewer than {MIN_COUNT} galaxies per bin',
             transform=axl.transAxes, fontsize=9.5, color='0.3', va='bottom')

    os.makedirs(outdir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(outdir, f'{OUTPUT_NAME}.{ext}')
        fig.savefig(path, dpi=170 if ext == 'png' else None,
                    bbox_inches='tight')
        print(f'  wrote {path}')
    plt.close(fig)


# ========================== TABLES ==========================

def _interp(x, y, x0):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    return float(np.interp(x0, x[good], y[good], left=np.nan, right=np.nan))


def write_tables(variants, results, sim, outdir):
    os.makedirs(outdir, exist_ok=True)
    ref = results[REFERENCE_KEY]
    cw = load_cosmos_web_csfrd()
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit('=' * 104)
    emit('SAGE26 cooling-sink series')
    emit('=' * 104)
    emit(f'volume {sim["volume"]:.4g} Mpc^3  (box {sim["box_size"]:g} Mpc/h, '
         f'h = {sim["hubble_h"]:g}, volume fraction {sim["volume_fraction"]:.3f})')
    emit(f'a bin holding {MIN_COUNT} galaxies sits at '
         f'log phi = {np.log10(MIN_COUNT / sim["volume"] / BINWIDTH):.2f}')
    emit()

    for tag, masses, title in (
            ('smf_z=0', TABLE_SMF_MASSES, 'stellar mass function, z = 0'),
            ('smf_z=2', TABLE_SMF_MASSES, 'stellar mass function, z = 2'),
            ('himf',    TABLE_GAS_MASSES, 'HI mass function, z = 0'),
            ('h2mf',    TABLE_GAS_MASSES, 'H2 mass function, z = 0')):
        emit(f'{title}: change in log10 phi vs the published model [dex]')
        emit('  ' + f'{"variant":<28s}' + ''.join(f'{m:>11.1f}' for m in masses)
             + '     N in those bins (published)')
        rm = ref[tag]
        counts = [int(rm['n'][np.argmin(np.abs(rm['x'] - m))]) if rm['x'] is not None
                  else 0 for m in masses]
        for v in variants:
            if v['key'] == REFERENCE_KEY:
                continue
            m = results[v['key']][tag]
            if m['x'] is None or rm['x'] is None:
                continue
            cells = []
            for mass in masses:
                i = np.argmin(np.abs(rm['x'] - mass))
                d = m['phi'][i] - rm['phi'][i]
                cells.append('       --  ' if not np.isfinite(d) else f'{d:>+11.2f}')
            emit('  ' + f'{v["plain"]:<28s}' + ''.join(cells)
                 + '     ' + ', '.join(str(c) for c in counts))
        emit()

    if cw is not None:
        zc, p50 = cw[0], cw[1]
        emit('cosmic SFR density: model - COSMOS-Web [dex]  '
             '(positive = too much star formation)')
        emit('  ' + f'{"variant":<28s}' + ''.join(f'{z:>11.1f}' for z in TABLE_REDSHIFTS))
        for v in variants:
            r = results[v['key']]['csfrd']
            cells = []
            for z in TABLE_REDSHIFTS:
                model = _interp(r['z'][::-1], r['phi'][::-1], z)
                obs = float(np.interp(z, zc, p50))
                cells.append('       --  ' if not np.isfinite(model)
                             else f'{model - obs:>+11.2f}')
            emit('  ' + f'{v["plain"]:<28s}' + ''.join(cells))
        emit()

    path = os.path.join(outdir, f'{OUTPUT_NAME}.txt')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  wrote {path}')


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true', help='run missing variants first')
    ap.add_argument('--force', action='store_true', help='re-run every variant')
    ap.add_argument('--only', default=None,
                    help='comma-separated subset of variant keys (base is always kept)')
    ap.add_argument('--outdir', default='./output/sink_series/')
    args = ap.parse_args()

    variants = VARIANTS
    if args.only:
        keep = {k.strip() for k in args.only.split(',')} | {REFERENCE_KEY}
        variants = [v for v in VARIANTS if v['key'] in keep]
        unknown = keep - {v['key'] for v in VARIANTS}
        if unknown:
            sys.exit(f'unknown variant keys: {", ".join(sorted(unknown))}')

    if args.run or args.force:
        print('Running variants:')
        run_variants(variants, force=args.force)
    else:
        for v in variants:
            write_variant_par(v)

    missing = [v['key'] for v in variants
               if not pp.model_files_exist(variant_out_dir(v['key']))]
    if missing:
        sys.exit('no output for: ' + ', '.join(missing) + '  -- pass --run')

    sim = read_sim(variant_out_dir(REFERENCE_KEY))
    if sim is None:
        sys.exit('could not read the reference run header')

    print('Measuring:')
    results = measure(variants, sim)
    write_tables(variants, results, sim, args.outdir)
    print('Plotting:')
    make_figure(variants, results, sim, args.outdir)


if __name__ == '__main__':
    main()

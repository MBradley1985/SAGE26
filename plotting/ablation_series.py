#!/usr/bin/env python
"""
SAGE26 ablation series
======================
Isolates the effect of each new SAGE26 ingredient by re-running the fiducial
model with that ingredient -- and only that ingredient -- switched off, then
plotting the stellar mass function and the cosmic star formation rate density
of every variant against the fiducial model.

Each ablation parameter file differs from ``input/millennium_all.par`` by a
single line (see the header of each file).  Nothing is recalibrated: the point
of the series is to isolate one ingredient at fixed calibration, so every other
parameter is held at its fiducial value.  A consequence worth stating in the
text is that the ablated runs are therefore *not* re-tuned models -- they show
what the fiducial calibration does without that piece of physics, which is the
quantity relevant to "what does this module contribute?".

Usage
-----
    python plotting/ablation_series.py            # plot from existing output
    python plotting/ablation_series.py --run      # run any missing variants first
    python plotting/ablation_series.py --run --force   # re-run every variant
    python plotting/ablation_series.py --with rps      # add an optional ablation
    python plotting/ablation_series.py --with rps --with snecons
    python plotting/ablation_series.py --no-sage16     # drop the SAGE16 reference

The figure covers the four ingredients the paper's headline claim rests on: FIRE
stellar feedback, H2-based star formation, the two-regime CGM and the FFB mode, plus
a joint run with all four disabled together.  Ram-pressure stripping and the SN energy
bound are available through --with but are off by default: neither is part of those
claims.

Must be run from the repository root (paths are relative, as in the .par files).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The paper module owns the simulation constants (volume, mass conversion,
# redshift table), the mass-function estimator and the observational
# compilations.  Reusing it keeps this figure on exactly the same footing as
# every other stellar mass function in the paper.
import paper_plots as pp


# ========================== CONFIGURATION ==========================

SAGE_BINARY = './sage'

# One entry per curve.  'par' is the parameter file, 'out' its OutputDir, and
# 'switch' the single parameter that differs from the fiducial run.
VARIANTS = [
    {'key': 'full',   'par': 'input/millennium_all.par',
     'out': './output/millennium/',
     'label': r'SAGE26 (fiducial)',            'switch': None,
     'color': 'black',   'ls': '-',  'lw': 3.6, 'zorder': 12},
    {'key': 'nofire', 'par': 'input/millennium_nofire.par',
     'out': './output/millennium_nofire/',
     'label': r'no FIRE feedback',             'switch': ('FIREmodeOn', 0),
     'color': '#0C5DA5', 'ls': (0, (6, 2)),       'lw': 2.4, 'zorder': 10},
    {'key': 'noh2',   'par': 'input/millennium_noh2.par',
     'out': './output/millennium_noh2/',
     'label': r'no H$_2$ star formation',      'switch': ('SFprescription', 0),
     'color': '#00B945', 'ls': (0, (1, 1.4)),     'lw': 2.6, 'zorder': 10},
    {'key': 'nocgm',  'par': 'input/millennium_nocgm.par',
     'out': './output/millennium_nocgm/',
     'label': r'no two-regime CGM',            'switch': ('CGMrecipeOn', 0),
     'color': '#FF9500', 'ls': (0, (7, 2, 1.5, 2)), 'lw': 2.4, 'zorder': 10},
    {'key': 'noffb',  'par': 'input/millennium_noffb.par',
     'out': './output/millennium_noffb/',
     'label': r'no FFB mode',                  'switch': ('FeedbackFreeModeOn', 0),
     'color': '#FF2C00', 'ls': (0, (3, 1.6)),     'lw': 2.4, 'zorder': 11},
    {'key': 'noallfour', 'par': 'input/millennium_noallfour.par',
     'out': './output/millennium_noallfour/',
     'label': r'all four removed',
     'switch': [('FIREmodeOn', 0), ('SFprescription', 0),
                ('CGMrecipeOn', 0), ('FeedbackFreeModeOn', 0)],
     'color': '#845B97', 'ls': '-',  'lw': 2.8, 'zorder': 8},
    {'key': 'sage16', 'par': 'input/millennium_vanilla.par',
     'out': './output/millennium_vanilla/',
     'label': r'SAGE16 (separately calibrated)', 'switch': None,
     'color': '#474747', 'ls': (0, (10, 3)),      'lw': 2.4, 'zorder': 9},
]

# The four ingredients whose individual contributions sum to the joint ablation.
# Comparing that sum against JOINT_KEY measures how far they are from acting
# independently, without the calibration differences that make SAGE16 unusable
# for the purpose.
FOUR_KEYS = ('nofire', 'noh2', 'nocgm', 'noffb')
JOINT_KEY = 'noallfour'

REFERENCE_KEY = 'full'      # residuals are measured against this variant

# Ingredients that are not part of the claims this figure supports are left out by
# default and switched on individually.  Ram-pressure stripping moves the HI content
# far more than it moves the stellar mass function; the SN energy bound postdates the
# submitted version, so it is not one of the four the referee asked about.
OPTIONAL_VARIANTS = {
    'rps': {
        'key': 'norps',  'par': 'input/millennium_norps.par',
        'out': './output/millennium_norps/',
        'label': r'no ram-pressure stripping', 'switch': ('RamPressureStrippingOn', 0),
        'color': '#8C564B', 'ls': (0, (5, 1.5, 1.5, 1.5, 1.5, 1.5)),
        'lw': 2.4, 'zorder': 10,
    },
    'snecons': {
        'key': 'nosnecons', 'par': 'input/millennium_nosnecons.par',
        'out': './output/millennium_nosnecons/',
        'label': r'no SN energy bound', 'switch': ('SNEnergyConservationOn', 0),
        'color': '#17A2B8', 'ls': (0, (2, 1, 5, 1)), 'lw': 2.4, 'zorder': 10,
    },
}

# Stellar mass function panels.  'select' is 'all', 'sf' or 'q': the sSFR-split
# panels use pp.SSFR_CUT, the same division as every other figure in the paper.
# The star-forming panel at cosmic noon is the one that tests the claim about the
# number density of massive *star-forming* galaxies at z ~ 2.
SMF_PANELS = [
    {'z': 0.0, 'select': 'all', 'tag': 'z=0',
     'xlim': (7.6, 12.4), 'ylim': (-6.2, -0.7)},
    {'z': 2.0, 'select': 'all', 'tag': 'z=2',
     'xlim': (7.6, 12.4), 'ylim': (-6.2, -0.7)},
    {'z': 2.0, 'select': 'sf',  'tag': 'z=2 star-forming',
     'xlim': (7.6, 12.4), 'ylim': (-6.2, -0.7)},
    {'z': 6.0, 'select': 'all', 'tag': 'z=6',
     'xlim': (7.6, 12.4), 'ylim': (-6.2, -0.7)},
]
SMF_BINWIDTH = 0.2
SMF_MASS_RANGE = (6.0, 13.0)    # shared bins so residuals are element-wise
SMF_OBS_DZ = 0.5                # observations within |z_obs - z_panel| of a panel
SMF_ROBUST_MIN_COUNT = 10       # ignore bins holding fewer galaxies when ranking
                                # offsets; the density floor follows from the volume

SELECT_LABEL = {'all': None, 'sf': 'star-forming', 'q': 'quiescent'}

RESIDUAL_YLIM = (-1.3, 1.3)
RESIDUAL_NEGLIGIBLE = 0.1   # shaded band marking differences below this, in dex

CSFRD_ZLIM = (0.0, 10.0)
CSFRD_YLIM = (-3.4, -0.4)

# Masses at which the printed table quotes residuals.
TABLE_MASSES = (8.5, 9.5, 10.5, 11.5)
TABLE_REDSHIFTS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)

OUTPUT_NAME = 'Ablation_Series'


# ========================== RUNNING THE MODEL ==========================

def run_variants(variants, force=False):
    """Execute SAGE for each variant whose output is missing (or all, if *force*)."""
    for v in variants:
        if not os.path.exists(v['par']):
            print(f"  {v['key']:>7s}: parameter file {v['par']} missing -- skipped")
            continue
        have_output = pp.model_files_exist(v['out'])
        if have_output and not force:
            print(f"  {v['key']:>7s}: output already present in {v['out']} -- skipped")
            continue
        os.makedirs(v['out'], exist_ok=True)
        print(f"  {v['key']:>7s}: {SAGE_BINARY} {v['par']}")
        res = subprocess.run([SAGE_BINARY, v['par']],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"    FAILED (exit {res.returncode})")
            sys.stderr.write(res.stderr.decode(errors='replace')[-2000:])
            sys.exit(res.returncode)
    print()


def read_par(path):
    """Parse a SAGE parameter file into {name: value}, dropping '%' comments."""
    params = {}
    if not os.path.exists(path):
        return params
    with open(path) as f:
        for line in f:
            line = line.split('%', 1)[0].strip()
            if not line or line.startswith('->'):
                continue
            fields = line.split()
            if len(fields) >= 2:
                params[fields[0]] = fields[1]
    return params


def check_switches(variants):
    """
    Confirm that each ablation run really differs from the fiducial run by its
    one advertised switch and nothing else.

    Two independent records are checked: the parameter files (what was asked
    for) and the ``Header/Runtime`` attributes of the output (what actually
    ran).  Not every parameter is written to the header -- those that are
    missing are reported rather than silently passed.

    Returns the list of variants that have usable output.
    """
    import h5py as h5

    available, runtime = [], {}
    for v in variants:
        files = pp.find_model_files(v['out'])
        if not files:
            print(f"  {v['key']:>7s}: no output in {v['out']} -- dropped from the figure")
            continue
        with h5.File(files[0], 'r') as f:
            runtime[v['key']] = dict(f['Header/Runtime'].attrs)
        available.append(v)

    ref_hdr = runtime.get(REFERENCE_KEY)
    if ref_hdr is None:
        print('  Warning: fiducial run unavailable, cannot verify switches')
        return available

    ref_variant = next(v for v in VARIANTS if v['key'] == REFERENCE_KEY)
    ref_par = read_par(ref_variant['par'])

    ignored = {'OutputDir', 'FileNameGalaxies'}
    unrecorded = set()

    for v in available:
        if v['key'] == REFERENCE_KEY:
            continue

        hdr = runtime[v['key']]
        hdr_diff = sorted(k for k in set(ref_hdr) | set(hdr)
                          if k not in ignored and ref_hdr.get(k) != hdr.get(k))

        par = read_par(v['par'])
        par_diff = sorted(k for k in set(ref_par) | set(par)
                          if k not in ignored and ref_par.get(k) != par.get(k))

        if v['switch'] is None:
            # A reference model rather than an ablation.  The header diff is
            # the meaningful one: its parameter file omits switches that then
            # fall back to the code defaults, which inflates the .par diff.
            print(f"  {v['key']:>9s}: reference model, not an ablation "
                  f"-- {len(hdr_diff)} recorded parameters differ: {hdr_diff}")
            continue

        # One switch or several (the joint ablation turns off all four).
        switches = v['switch'] if isinstance(v['switch'], list) else [v['switch']]
        expected = sorted(name for name, _ in switches)
        missing = [name for name, _ in switches
                   if not (name in ref_hdr and name in hdr)]
        unrecorded.update(missing)

        values_ok = all(par.get(name) == str(value) for name, value in switches)
        hdr_expected = sorted(n for n in expected if n not in missing)
        ok = (par_diff == expected and values_ok and hdr_diff == hdr_expected)
        note = '' if not missing \
            else f'  ({", ".join(missing)} not written to the HDF5 header)'
        setting = ', '.join(f'{name}={par.get(name)}' for name, _ in switches)
        print(f"  {v['key']:>9s}: {setting};  .par diff: {par_diff};  "
              f"header diff: {hdr_diff}  [{'OK' if ok else 'CHECK'}]{note}")

    if unrecorded:
        print(f"\n  Note: {', '.join(sorted(unrecorded))} "
              f"{'is' if len(unrecorded) == 1 else 'are'} absent from "
              f"Header/Runtime in the output files, so "
              f"{'that switch' if len(unrecorded) == 1 else 'those switches'} "
              f"could only be verified from the parameter files.")
    print()
    return available


# ========================== MEASUREMENTS ==========================

def read_sim(directory):
    """
    Volume, mass conversion and redshift table of the run in *directory*, taken
    from its own HDF5 header rather than from the module-level constants in
    ``paper_plots`` (which describe mini-Millennium only).  Without this the
    series would report Millennium volumes and redshifts for any other
    simulation, silently and with no error.
    """
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


def check_same_simulation(sim, variants):
    """
    Warn if any variant was run on a different simulation from the reference.
    Mixing volumes or redshift tables in one ablation series would make the
    residuals meaningless, so it is reported rather than absorbed.
    """
    mismatched = []
    for v in variants:
        other = read_sim(v['out'])
        if other is None:
            continue
        same = (np.isclose(other['box_size'], sim['box_size'])
                and np.isclose(other['hubble_h'], sim['hubble_h'])
                and np.isclose(other['volume_fraction'], sim['volume_fraction'])
                and other['redshifts'].size == sim['redshifts'].size
                and np.allclose(other['redshifts'], sim['redshifts']))
        if not same:
            mismatched.append(v['key'])
            print(f"  {v['key']:>9s}: WARNING -- different simulation "
                  f"(box {other['box_size']:g} Mpc/h, h = {other['hubble_h']:g}, "
                  f"{other['redshifts'].size} snapshots) than the reference run "
                  f"(box {sim['box_size']:g} Mpc/h, h = {sim['hubble_h']:g}, "
                  f"{sim['redshifts'].size} snapshots)")
    if not mismatched:
        print(f"  all runs on the same simulation: box {sim['box_size']:g} Mpc/h, "
              f"h = {sim['hubble_h']:g}, {sim['redshifts'].size} snapshots, "
              f"volume {sim['volume']:.3g} Mpc^3")
    print()
    return mismatched


def density_floor(sim, min_count=None, binwidth=None):
    """
    log10 phi below which a bin holds fewer than *min_count* galaxies.

    Scales with the volume, so a larger box automatically pushes the floor down
    instead of leaving a hardcoded threshold that was tuned for one box.
    """
    min_count = SMF_ROBUST_MIN_COUNT if min_count is None else min_count
    binwidth = SMF_BINWIDTH if binwidth is None else binwidth
    return float(np.log10(min_count / sim['volume'] / binwidth))


def smf(path, z_target, sim, select='all'):
    """
    Stellar mass function of *path* at the output snapshot nearest *z_target*.

    *select* is 'all', 'sf' (log sSFR > pp.SSFR_CUT) or 'q' (below the cut).

    Returns (snapshot number, snapshot redshift, bin centres, log10 phi).
    """
    redshifts = sim['redshifts']
    snap = pp._snap_nearest_z(redshifts, z_target)
    props = ['StellarMass']
    if select != 'all':
        props += ['SfrDisk', 'SfrBulge']
    data = pp.read_snap_from_files(pp.find_model_files(path), f'Snap_{snap}',
                                   props, mass_convert=sim['mass_convert'])
    if not data:
        return snap, np.nan, None, None

    m = data['StellarMass']
    keep = m > 0
    if select != 'all':
        # Galaxies with zero star formation have log sSFR = -inf, so they fall
        # on the quiescent side of the cut rather than being dropped.
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = pp.log_ssfr(data['SfrDisk'], data['SfrBulge'], m)
        keep &= (ssfr > pp.SSFR_CUT) if select == 'sf' else (ssfr <= pp.SSFR_CUT)

    m = m[keep]
    if m.size == 0:
        return snap, redshifts[snap], None, None
    x, phi, _ = pp.mass_function(np.log10(m), sim['volume'],
                                binwidth=SMF_BINWIDTH, mass_range=SMF_MASS_RANGE)
    return snap, redshifts[snap], x, phi


def csfrd(path, sim):
    """
    Cosmic star formation rate density over every output snapshot.

    Returns (redshifts, log10 rho_SFR) with NaN where a snapshot is empty.
    """
    files = pp.find_model_files(path)
    z = sim['redshifts']
    rho = np.full(z.size, np.nan)
    for snap in range(z.size):
        d = pp.read_snap_from_files(files, f'Snap_{snap}', ['SfrDisk', 'SfrBulge'],
                                    mass_convert=sim['mass_convert'])
        if not d:
            continue
        total = np.sum(d['SfrDisk'] + d['SfrBulge'])
        if total > 0:
            rho[snap] = total / sim['volume']
    with np.errstate(divide='ignore', invalid='ignore'):
        return z, np.log10(rho)


def panel_id(panel):
    """Key for a panel's measurement: two panels may share a redshift."""
    return (panel['z'], panel['select'])


def measure(variants, sim):
    """Compute every SMF panel and the CSFRD for every variant."""
    out = {}
    for v in variants:
        print(f"  {v['key']:>9s}: {v['out']}")
        entry = {'smf': {}}
        for panel in SMF_PANELS:
            snap, z_snap, x, phi = smf(v['out'], panel['z'], sim, panel['select'])
            entry['smf'][panel_id(panel)] = {'snap': snap, 'z': z_snap,
                                             'x': x, 'phi': phi}
        entry['z'], entry['csfrd'] = csfrd(v['out'], sim)
        out[v['key']] = entry
    print()
    return out


# ========================== PLOTTING ==========================

def load_muzzin13_split(select, z_target, hubble_h):
    """
    Muzzin et al. (2013) sSFR-split stellar mass functions -- the paper's
    all/quiescent/star-forming compilation, whose UVJ split is the observational
    counterpart of the model sSFR cut.

    Columns are z_lo z_hi logM E_logM then (logPhi, EU, EL) for all, quiescent
    and star-forming in turn, with -99 marking bins without a measurement.
    Returns one dict per redshift bin overlapping *z_target*, in the same shape
    as ``pp._load_smf_grid_observations`` entries.
    """
    path = './data/smf/SMF_Muzzin2013.dat'
    if not os.path.exists(path) or select == 'all':
        return []
    col = {'q': 7, 'sf': 10}[select]
    h_m = 0.7                                  # their h, converted to ours
    log_phi_corr = 3.0 * np.log10(h_m / hubble_h)

    bins = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) < col + 3:
                continue
            z_lo, z_hi = float(p[0]), float(p[1])
            if not z_lo - 1e-6 <= z_target <= z_hi + 1e-6:
                continue
            log_m, log_phi = float(p[2]), float(p[col])
            e_hi, e_lo = float(p[col + 1]), float(p[col + 2])
            if log_phi < -10 or not np.isfinite(log_phi):
                continue
            b = bins.setdefault((z_lo, z_hi), {'m': [], 'lp': [], 'eu': [], 'el': []})
            b['m'].append(log_m - 0.04)        # Kroupa -> Chabrier
            b['lp'].append(log_phi + log_phi_corr)
            b['eu'].append(max(e_hi, 0.0))
            b['el'].append(max(e_lo, 0.0))

    out = []
    for (z_lo, z_hi), b in sorted(bins.items()):
        out.append({'z': 0.5 * (z_lo + z_hi),
                    'log_mass': np.array(b['m']), 'log_phi': np.array(b['lp']),
                    'err_lo': np.array(b['el']), 'err_hi': np.array(b['eu']),
                    'label': f'Muzzin+13 ({SELECT_LABEL[select]})',
                    'marker': '^', 'ms': 8})
    return out


def _draw_smf_observations(ax, all_obs, z_panel, seen_labels):
    """Overlay every observational SMF within SMF_OBS_DZ of *z_panel*."""
    drawn = []
    for od in all_obs:
        if abs(od['z'] - z_panel) > SMF_OBS_DZ:
            continue
        yerr = None
        if od['err_lo'] is not None and od['err_hi'] is not None:
            yerr = [od['err_lo'], od['err_hi']]
        label = None
        if 'obs' not in seen_labels:
            label = 'observations'
            seen_labels.add('obs')
        ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                    fmt=od['marker'], color='grey', ms=od['ms'],
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor='gray', alpha=0.55, lw=1.0,
                    label=label, zorder=2)
        drawn.append(f"{od['label']} (z={od['z']:g})")
    return drawn


def _draw_csfrd_observations(ax):
    """Somerville+01 compilation and the Madau & Dickinson (2014) fit."""
    obs = np.array([
        [0, 0.0158489, 0, 0, 0.0251189, 0.01000000],
        [0.150000, 0.0173780, 0, 0.300000, 0.0181970, 0.0165959],
        [0.0425000, 0.0239883, 0.0425000, 0.0425000, 0.0269153, 0.0213796],
        [0.200000, 0.0295121, 0.100000, 0.300000, 0.0323594, 0.0269154],
        [0.350000, 0.0147911, 0.200000, 0.500000, 0.0173780, 0.0125893],
        [0.625000, 0.0275423, 0.500000, 0.750000, 0.0331131, 0.0229087],
        [0.825000, 0.0549541, 0.750000, 1.00000, 0.0776247, 0.0389045],
        [0.625000, 0.0794328, 0.500000, 0.750000, 0.0954993, 0.0660693],
        [0.700000, 0.0323594, 0.575000, 0.825000, 0.0371535, 0.0281838],
        [1.25000, 0.0467735, 1.50000, 1.00000, 0.0660693, 0.0331131],
        [0.750000, 0.0549541, 0.500000, 1.00000, 0.0389045, 0.0776247],
        [1.25000, 0.0741310, 1.00000, 1.50000, 0.0524807, 0.104713],
        [1.75000, 0.0562341, 1.50000, 2.00000, 0.0398107, 0.0794328],
        [2.75000, 0.0794328, 2.00000, 3.50000, 0.0562341, 0.112202],
        [4.00000, 0.0309030, 3.50000, 4.50000, 0.0489779, 0.0194984],
        [0.250000, 0.0398107, 0.00000, 0.500000, 0.0239883, 0.0812831],
        [0.750000, 0.0446684, 0.500000, 1.00000, 0.0323594, 0.0776247],
        [1.25000, 0.0630957, 1.00000, 1.50000, 0.0478630, 0.109648],
        [1.75000, 0.0645654, 1.50000, 2.00000, 0.0489779, 0.112202],
        [2.50000, 0.0831764, 2.00000, 3.00000, 0.0512861, 0.158489],
        [3.50000, 0.0776247, 3.00000, 4.00000, 0.0416869, 0.169824],
        [4.50000, 0.0977237, 4.00000, 5.00000, 0.0416869, 0.269153],
        [5.50000, 0.0426580, 5.00000, 6.00000, 0.0177828, 0.165959],
        [3.00000, 0.120226, 2.00000, 4.00000, 0.173780, 0.0831764],
        [3.04000, 0.128825, 2.69000, 3.39000, 0.151356, 0.109648],
        [4.13000, 0.114815, 3.78000, 4.48000, 0.144544, 0.0912011],
        [0.350000, 0.0346737, 0.200000, 0.500000, 0.0537032, 0.0165959],
        [0.750000, 0.0512861, 0.500000, 1.00000, 0.0575440, 0.0436516],
        [1.50000, 0.0691831, 1.00000, 2.00000, 0.0758578, 0.0630957],
        [2.50000, 0.147911, 2.00000, 3.00000, 0.169824, 0.128825],
        [3.50000, 0.0645654, 3.00000, 4.00000, 0.0776247, 0.0512861],
    ], dtype=np.float64)
    z = obs[:, 0]
    log_rho = np.log10(obs[:, 1])
    ax.errorbar(z, log_rho,
                yerr=[np.abs(log_rho - np.log10(obs[:, 4])),
                      np.abs(np.log10(obs[:, 5]) - log_rho)],
                xerr=[np.abs(obs[:, 0] - obs[:, 2]), np.abs(obs[:, 3] - obs[:, 0])],
                fmt='o', markerfacecolor='gray', markeredgecolor='k',
                markeredgewidth=1.0, ecolor='k', color='k', ms=7, lw=1.0,
                alpha=0.55, ls='none', zorder=2, label='observations')

    # Madau & Dickinson (2014), Chabrier -> Salpeter (factor 1/0.63) to match SAGE.
    zz = np.linspace(CSFRD_ZLIM[0], CSFRD_ZLIM[1], 300)
    psi = 0.015 * (1 + zz)**2.7 / (1 + ((1 + zz) / 2.9)**5.6) / 0.63
    ax.plot(zz, np.log10(psi), color='gray', lw=1.5, alpha=0.7, zorder=2,
            label=pp._tex_safe(r'Madau \& Dickinson 2014'))


def make_figure(variants, results, sim, outdir):
    """Two-row figure: absolute measurements on top, residuals below."""
    ncols = len(SMF_PANELS) + 1
    fig = plt.figure(figsize=(5.6 * ncols, 9.6))
    fig.set_tight_layout(False)
    gs = fig.add_gridspec(2, ncols, height_ratios=[2.05, 1.0],
                          hspace=0.06, wspace=0.28)

    ref = results[REFERENCE_KEY]
    all_obs = pp._load_smf_grid_observations()
    seen_labels = set()
    obs_used = {}

    panel_letters = 'abcdefgh'

    # ---- stellar mass function columns ----
    for col, panel in enumerate(SMF_PANELS):
        z_panel, select = panel['z'], panel['select']
        pid = panel_id(panel)
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)

        # The all-galaxy compilation does not apply to an sSFR-split panel, so
        # those panels take only the matching split from Muzzin+13.
        panel_obs = (all_obs if select == 'all'
                     else load_muzzin13_split(select, z_panel, sim['hubble_h']))
        obs_used[panel['tag']] = _draw_smf_observations(
            ax, panel_obs, z_panel, seen_labels)

        ref_phi = ref['smf'][pid]['phi']
        for v in variants:
            m = results[v['key']]['smf'][pid]
            if m['phi'] is None:
                continue
            good = np.isfinite(m['phi'])
            ax.plot(m['x'][good], m['phi'][good], color=v['color'], ls=v['ls'],
                    lw=v['lw'], zorder=v['zorder'], label=v['label'])
            if v['key'] == REFERENCE_KEY or ref_phi is None:
                continue
            delta = m['phi'] - ref_phi
            good = np.isfinite(delta)
            axr.plot(m['x'][good], delta[good], color=v['color'], ls=v['ls'],
                     lw=v['lw'], zorder=v['zorder'])

        z_snap = ref['smf'][pid]['z']
        sel_label = SELECT_LABEL[select]
        annotation = rf'$z = {z_snap:.2f}$'
        if sel_label is not None:
            # Plain text rather than \mathrm{}: the hyphen in "star-forming"
            # renders as a minus sign in math mode.
            annotation += '\n' + sel_label
        ax.text(0.05, 0.06, annotation, transform=ax.transAxes,
                ha='left', va='bottom')
        ax.text(0.04, 0.95, rf'$\mathrm{{({panel_letters[col]})}}$',
                transform=ax.transAxes, ha='left', va='top', fontsize=15)

        ax.set_xlim(*panel['xlim'])
        ax.set_ylim(*panel['ylim'])
        _residual_guides(axr)
        axr.set_ylim(*RESIDUAL_YLIM)
        axr.set_xlabel(r'$\log_{10}\ m_{*}\ [M_{\odot}]$')
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
            axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')
        _format(ax, xmaj=1.0, xmin=0.2, ymaj=1.0, ymin=0.2, hide_xticklabels=True)
        _format(axr, xmaj=1.0, xmin=0.2, ymaj=0.5, ymin=0.1)

    # ---- CSFRD column ----
    ax = fig.add_subplot(gs[0, ncols - 1])
    axr = fig.add_subplot(gs[1, ncols - 1], sharex=ax)
    _draw_csfrd_observations(ax)

    ref_rho = ref['csfrd']
    for v in variants:
        r = results[v['key']]
        good = np.isfinite(r['csfrd'])
        ax.plot(r['z'][good], r['csfrd'][good], color=v['color'], ls=v['ls'],
                lw=v['lw'], zorder=v['zorder'], label=v['label'])
        if v['key'] == REFERENCE_KEY:
            continue
        delta = r['csfrd'] - ref_rho
        good = np.isfinite(delta)
        axr.plot(r['z'][good], delta[good], color=v['color'], ls=v['ls'],
                 lw=v['lw'], zorder=v['zorder'])

    ax.text(0.04, 0.95, rf'$\mathrm{{({panel_letters[ncols - 1]})}}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=15)
    ax.set_xlim(*CSFRD_ZLIM)
    ax.set_ylim(*CSFRD_YLIM)
    ax.set_ylabel(r'$\log_{10}\ \rho_{\rm SFR}\ '
                  r'[M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3}]$')
    _residual_guides(axr)
    axr.set_ylim(*RESIDUAL_YLIM)
    axr.set_xlabel(r'$\mathrm{Redshift}$')
    axr.set_ylabel(r'$\Delta \log_{10}\ \rho_{\rm SFR}$')
    _format(ax, xmaj=2.0, xmin=0.5, ymaj=1.0, ymin=0.2, hide_xticklabels=True)
    _format(axr, xmaj=2.0, xmin=0.5, ymaj=0.5, ymin=0.1)

    # ---- legends: models above the figure, observations in the CSFRD panel ----
    # A figure-level legend keeps the model curves unobscured; with seven
    # variants no panel has room for it.
    model_labels = [v['label'] for v in variants]
    handles, labels = fig.axes[0].get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l in model_labels]
    order = {v['label']: i for i, v in enumerate(variants)}
    keep.sort(key=lambda hl: order[hl[1]])
    fig.legend([h for h, _ in keep], [l for _, l in keep],
               loc='lower center', bbox_to_anchor=(0.5, 0.905),
               ncol=len(keep), frameon=False, fontsize=15,
               handlelength=3.0, columnspacing=1.6)

    handles, labels = ax.get_legend_handles_labels()
    obs_keep = [(h, l) for h, l in zip(handles, labels) if l not in model_labels]
    if obs_keep:
        ax.legend([h for h, _ in obs_keep], [l for _, l in obs_keep],
                  loc='lower left', frameon=False, fontsize=13, labelspacing=0.25)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, OUTPUT_NAME + pp.OUTPUT_FORMAT)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    for tag, used in obs_used.items():
        print(f'  {tag} observations: '
              + (', '.join(sorted(set(used))) if used else 'none available'))
    print()
    return path


def _residual_guides(ax):
    """Zero line plus a band marking differences too small to matter."""
    ax.axhspan(-RESIDUAL_NEGLIGIBLE, RESIDUAL_NEGLIGIBLE,
               color='0.85', alpha=0.6, lw=0, zorder=0)
    ax.axhline(0.0, color='black', lw=1.0, ls='-', alpha=0.6, zorder=1)


def _format(ax, xmaj, xmin, ymaj, ymin, hide_xticklabels=False):
    ax.xaxis.set_major_locator(plt.MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(xmin))
    ax.yaxis.set_major_locator(plt.MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(ymin))
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    if hide_xticklabels:
        ax.tick_params(labelbottom=False)


# ========================== TABLES ==========================

def _interp(x, y, x0):
    """Linear interpolation of *y* onto *x0*, ignoring non-finite samples."""
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    xs, ys = np.asarray(x)[good], np.asarray(y)[good]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x0 < xs[0] or x0 > xs[-1]:
        return np.nan
    return float(np.interp(x0, xs, ys))


def _cell(value, delta=None, width=15):
    """Format one table cell as 'value' or 'value (+delta)'."""
    if not np.isfinite(value):
        return f'{"--":>{width}s}'
    if delta is None:
        return f'{value:>{width}.2f}'
    if not np.isfinite(delta):
        return f'{value:>{width}.2f}'
    return f'{f"{value:.2f} ({delta:+.2f})":>{width}s}'


def _largest_deviation(x, delta, reference, xlim=None, floor=None):
    """
    Location and size of the largest |delta|, restricted to where the
    measurement is trustworthy: inside *xlim*, and above the *floor* in the
    reference curve so that Poisson noise in near-empty bins is not reported
    as the dominant effect.
    """
    x = np.asarray(x, dtype=float)
    good = np.isfinite(delta) & np.isfinite(x) & np.isfinite(reference)
    if xlim is not None:
        good &= (x >= xlim[0]) & (x <= xlim[1])
    if floor is not None:
        good &= reference >= floor
    if not np.any(good):
        return np.nan, np.nan
    idx = np.nanargmax(np.abs(np.where(good, delta, np.nan)))
    return x[idx], delta[idx]


def write_tables(variants, results, sim, outdir):
    """
    Print, and save, the numbers behind every panel of the figure.

    Each cell gives the plotted quantity and, in brackets, its offset from the
    fiducial run -- so the top and bottom rows of the figure can both be read
    off the table.
    """
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    ref = results[REFERENCE_KEY]
    others = [v for v in variants if v['key'] != REFERENCE_KEY]
    letters = 'abcdefgh'
    summary = {v['key']: [] for v in others}
    floor = density_floor(sim)

    # ---- stellar mass function panels ----
    for col, panel in enumerate(SMF_PANELS):
        pid = panel_id(panel)
        ref_m = ref['smf'][pid]
        sel_label = SELECT_LABEL[panel['select']]
        which = 'stellar mass function' if sel_label is None \
            else f'{sel_label} stellar mass function'
        emit()
        emit('=' * 96)
        emit(f'PANEL ({letters[col]})   {which} at z = {ref_m["z"]:.2f}'
             f'  (snapshot {ref_m["snap"]})')
        emit('   log10 phi [Mpc^-3 dex^-1], with (variant - fiducial) in dex')
        if sel_label is not None:
            emit(f'   split at log sSFR = {pp.SSFR_CUT:.1f}')
        emit('=' * 96)
        emit('  ' + f'{"variant":<26s}' +
             ''.join(f'{f"logM*={m:.1f}":>15s}' for m in TABLE_MASSES) +
             f'{"largest offset":>26s}')

        ref_x, ref_phi = ref_m['x'], ref_m['phi']
        if ref_phi is None:
            emit('  fiducial unavailable at this redshift')
            continue

        emit('  ' + f'{"full (fiducial)":<26s}' +
             ''.join(_cell(_interp(ref_x, ref_phi, m)) for m in TABLE_MASSES))

        for v in others:
            m = results[v['key']]['smf'][pid]
            if m['phi'] is None:
                continue
            delta = m['phi'] - ref_phi
            cells = ''.join(
                _cell(_interp(m['x'], m['phi'], mass),
                      _interp(ref_x, delta, mass)) for mass in TABLE_MASSES)
            at, worst = _largest_deviation(ref_x, delta, ref_phi,
                                           xlim=panel['xlim'],
                                           floor=floor)
            note = ('--' if not np.isfinite(worst)
                    else f'{worst:+.2f} dex at logM*={at:.1f}')
            summary[v['key']].append(note)
            emit('  ' + f'{v["key"]:<26s}' + cells + f'{note:>26s}')

    # ---- how much the sSFR split actually changes ----
    # If nearly every massive galaxy is star-forming, the split panel repeats the
    # all-galaxy panel, and a claim about massive *star-forming* galaxies rests on
    # the same measurement as the claim about massive galaxies.
    for panel in SMF_PANELS:
        if panel['select'] == 'all':
            continue
        twin = (panel['z'], 'all')
        pid = panel_id(panel)
        if twin not in ref['smf'] or ref['smf'][twin]['phi'] is None:
            continue
        sel_label = SELECT_LABEL[panel['select']]
        emit()
        emit('=' * 96)
        emit(f'{sel_label.upper()} FRACTION at z = {ref["smf"][pid]["z"]:.2f}'
             f'   (phi_{panel["select"]} / phi_all, per cent)')
        emit('=' * 96)
        emit('  ' + f'{"variant":<26s}' +
             ''.join(f'{f"logM*={m:.1f}":>15s}' for m in TABLE_MASSES))
        for v in variants:
            r = results[v['key']]['smf']
            if r[pid]['phi'] is None or r[twin]['phi'] is None:
                continue
            cells = ''
            for mass in TABLE_MASSES:
                num = _interp(r[pid]['x'], r[pid]['phi'], mass)
                den = _interp(r[twin]['x'], r[twin]['phi'], mass)
                if np.isfinite(num) and np.isfinite(den):
                    cells += f'{100.0 * 10**(num - den):>15.1f}'
                else:
                    cells += f'{"--":>15s}'
            name = 'full (fiducial)' if v['key'] == REFERENCE_KEY else v['key']
            emit('  ' + f'{name:<26s}' + cells)

    # ---- CSFRD panel ----
    col = len(SMF_PANELS)
    emit()
    emit('=' * 96)
    emit(f'PANEL ({letters[col]})   cosmic star formation rate density')
    emit('   log10 rho_SFR [Msun yr^-1 Mpc^-3], with (variant - fiducial) in dex')
    emit('=' * 96)
    emit('  ' + f'{"variant":<26s}' +
         ''.join(f'{f"z={z:.0f}":>15s}' for z in TABLE_REDSHIFTS) +
         f'{"largest offset":>26s}')

    ref_z, ref_rho = ref['z'], ref['csfrd']
    emit('  ' + f'{"full (fiducial)":<26s}' +
         ''.join(_cell(_interp(ref_z, ref_rho, z)) for z in TABLE_REDSHIFTS))

    for v in others:
        r = results[v['key']]
        delta = r['csfrd'] - ref_rho
        cells = ''.join(
            _cell(_interp(r['z'], r['csfrd'], z), _interp(r['z'], delta, z))
            for z in TABLE_REDSHIFTS)
        at, worst = _largest_deviation(r['z'], delta, ref_rho, xlim=CSFRD_ZLIM)
        note = '--' if not np.isfinite(worst) else f'{worst:+.2f} dex at z={at:.1f}'
        summary[v['key']].append(note)
        emit('  ' + f'{v["key"]:<26s}' + cells + f'{note:>26s}')

    # ---- peak of the CSFRD ----
    emit()
    emit('=' * 96)
    emit('PEAK OF THE CSFRD')
    emit('=' * 96)
    ref_peak = None
    for v in variants:
        r = results[v['key']]
        good = np.isfinite(r['csfrd']) & (r['z'] <= CSFRD_ZLIM[1])
        if not np.any(good):
            continue
        idx = np.argmax(r['csfrd'][good])
        peak, z_peak = r['csfrd'][good][idx], r['z'][good][idx]
        if v['key'] == REFERENCE_KEY:
            ref_peak = (peak, z_peak)
            emit(f'  {v["key"]:<26s} {peak:+.2f} dex at z = {z_peak:.2f}')
        else:
            dz = z_peak - ref_peak[1] if ref_peak else np.nan
            dp = peak - ref_peak[0] if ref_peak else np.nan
            emit(f'  {v["key"]:<26s} {peak:+.2f} dex at z = {z_peak:.2f}'
                 f'   ({dp:+.2f} dex, dz = {dz:+.2f})')

    # ---- do the four ingredients act independently? ----
    have = {v['key'] for v in variants}
    if JOINT_KEY in have and set(FOUR_KEYS) <= have:
        emit()
        emit('=' * 96)
        emit('ARE THE FOUR INGREDIENTS INDEPENDENT?')
        emit(f'   sum      = {" + ".join(FOUR_KEYS)}, each measured on its own')
        emit(f'   joint    = {JOINT_KEY} (all four off in one run, nothing else changed)')
        emit('   residual = joint - sum. Zero means the ingredients act independently;')
        emit('              a non-zero residual is the interaction between them.')
        emit('=' * 96)

        def additivity(grid, offsets, joint, targets, sage16=None):
            """
            Summed offsets, joint offset and their residual; plus SAGE16 where
            available, which tests whether SAGE16 is a fair stand-in for
            "all four off" or is displaced by its separate calibration.
            """
            total = np.zeros_like(joint)
            for d in offsets:
                total = total + d
            rows = [('sum of the four', total), ('joint (all four off)', joint),
                    ('interaction residual', joint - total)]
            if sage16 is not None:
                rows += [('SAGE16, for comparison', sage16),
                         ('joint - SAGE16', joint - sage16)]
            for name, series in rows:
                cells = ''.join(_cell(_interp(grid, series, t)) for t in targets)
                emit('  ' + f'{name:<26s}' + cells)

        for col, panel in enumerate(SMF_PANELS):
            pid = panel_id(panel)
            ref_phi = ref['smf'][pid]['phi']
            joint_phi = results[JOINT_KEY]['smf'][pid]['phi']
            if ref_phi is None or joint_phi is None:
                continue
            offsets = [results[k]['smf'][pid]['phi'] - ref_phi for k in FOUR_KEYS
                       if results[k]['smf'][pid]['phi'] is not None]
            if len(offsets) != len(FOUR_KEYS):
                continue
            sel_label = SELECT_LABEL[panel['select']]
            title = f'panel ({letters[col]})  z = {ref["smf"][pid]["z"]:.2f}'
            if sel_label is not None:
                title += f', {sel_label}'
            emit()
            emit(f'  {title}   [dex]')
            emit('  ' + f'{"":<26s}' +
                 ''.join(f'{f"logM*={m:.1f}":>15s}' for m in TABLE_MASSES))
            s16 = None
            if 'sage16' in have:
                s16_phi = results['sage16']['smf'][pid]['phi']
                if s16_phi is not None:
                    s16 = s16_phi - ref_phi
            additivity(ref['smf'][pid]['x'], offsets,
                       joint_phi - ref_phi, TABLE_MASSES, s16)

        offsets = [results[k]['csfrd'] - ref_rho for k in FOUR_KEYS]
        emit()
        emit(f'  panel ({letters[len(SMF_PANELS)]})  cosmic SFR density   [dex]')
        emit('  ' + f'{"":<26s}' +
             ''.join(f'{f"z={z:.0f}":>15s}' for z in TABLE_REDSHIFTS))
        s16 = (results['sage16']['csfrd'] - ref_rho) if 'sage16' in have else None
        additivity(ref_z, offsets, results[JOINT_KEY]['csfrd'] - ref_rho,
                   TABLE_REDSHIFTS, s16)

    # ---- one-line-per-ingredient summary ----
    headers = []
    for p in SMF_PANELS:
        z_snap = ref['smf'][panel_id(p)]['z']
        sel = SELECT_LABEL[p['select']]
        headers.append(f'z={z_snap:.1f} SMF' if sel is None
                       else f'z={z_snap:.1f} SMF ({sel})')
    headers.append('CSFRD')
    emit()
    emit('=' * 96)
    emit('WHAT EACH INGREDIENT CONTRIBUTES  (largest offset from the fiducial run, '
         'per panel)')
    emit('=' * 96)
    emit('  ' + f'{"ingredient removed":<26s}' +
         ''.join(f'{h:>26s}' for h in headers))
    for v in others:
        emit('  ' + f'{v["key"]:<26s}' +
             ''.join(f'{s:>26s}' for s in summary[v['key']]))

    emit()
    emit(f'Simulation: box {sim["box_size"]:g} Mpc/h, h = {sim["hubble_h"]:g}, '
         f'volume {sim["volume"]:.3g} Mpc^3.')
    emit(f'The residual panels shade |offset| < {RESIDUAL_NEGLIGIBLE:.2f} dex: an ingredient whose '
         f'curve stays')
    emit('inside that band does not shape that measurement. "largest offset" ignores')
    emit(f'stellar mass bins holding fewer than {SMF_ROBUST_MIN_COUNT} galaxies in the '
         f'fiducial run, which')
    emit(f'for this volume and a {SMF_BINWIDTH:g} dex bin means log10 phi < {floor:.2f}. '
         f'A larger box lowers')
    emit('that floor and lets the massive end be quoted further out.')
    emit()
    emit('Note: no variant is recalibrated -- each shows the fiducial calibration')
    emit('with one ingredient removed. SAGE16 is a separately calibrated model, not')
    emit('a single-switch ablation, and is shown for reference only.')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, OUTPUT_NAME + '_stats.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Saved: {path}')
    return path


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true',
                    help='run SAGE for any variant whose output is missing')
    ap.add_argument('--force', action='store_true',
                    help='with --run, re-run every variant even if output exists')
    ap.add_argument('--with', dest='extra', action='append', default=[],
                    choices=sorted(OPTIONAL_VARIANTS), metavar='NAME',
                    help='also show an optional ablation, repeatable: '
                         + ', '.join(sorted(OPTIONAL_VARIANTS)))
    ap.add_argument('--with-rps', action='store_true',
                    help='shorthand for --with rps')
    ap.add_argument('--no-sage16', action='store_true',
                    help='omit the SAGE16 reference curve')
    ap.add_argument('--outdir', default=None,
                    help='where to write the figure (default: <fiducial>/plots/)')
    args = ap.parse_args()

    variants = [v for v in VARIANTS
                if not (args.no_sage16 and v['key'] == 'sage16')]
    extra = list(args.extra) + (['rps'] if args.with_rps else [])
    for name in dict.fromkeys(extra):
        # Before the joint and reference curves, which are drawn last.
        at = next((i for i, v in enumerate(variants)
                   if v['key'] in (JOINT_KEY, 'sage16')), len(variants))
        variants.insert(at, OPTIONAL_VARIANTS[name])

    if args.run:
        print('Running SAGE:')
        run_variants(variants, force=args.force)

    print('Verifying that each ablation differs by one switch:')
    variants = check_switches(variants)
    ref_variant = next((v for v in variants if v['key'] == REFERENCE_KEY), None)
    if ref_variant is None:
        sys.exit('Fiducial run not found; nothing to compare against.')

    # Every measurement uses the reference run's own simulation parameters, so
    # the series is correct on any box, not just the one paper_plots defaults to.
    print('Simulation:')
    sim = read_sim(ref_variant['out'])
    if sim is None:
        sys.exit(f'Could not read a simulation header from {ref_variant["out"]}.')
    check_same_simulation(sim, variants)

    print('Measuring:')
    results = measure(variants, sim)

    outdir = args.outdir or os.path.join(ref_variant['out'], 'plots/')
    np.random.seed(pp.SEED)
    pp.setup_style()

    print('Plotting:')
    make_figure(variants, results, sim, outdir)
    write_tables(variants, results, sim, outdir)


if __name__ == '__main__':
    main()

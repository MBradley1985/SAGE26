#!/usr/bin/env python
"""
allresults-compare.py -- allresults-local and allresults-history for several models at once.

Every model listed in MODELS below is drawn on the same figures: as one line each on the
line plots, and as one set of points each (or one panel each, where the points are already
coloured by a third quantity) on the scatter plots. The particle-length, Mvir and sSFR cuts
and the model colours are all set in the USER OPTIONS block.

Usage:
    python plotting/allresults-compare.py              # local + history plots
    python plotting/allresults-compare.py --local      # z=0 (single-snapshot) plots only
    python plotting/allresults-compare.py --history    # redshift-evolution plots only

Reads:  <dir>/model_*.hdf5 for each entry in MODELS
Writes: OUTPUT_DIR/*.png (one file per diagnostic figure)
"""

import argparse
import glob
import os
import re
import sys
import warnings

import h5py as h5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import interpolate
from scipy.stats import gaussian_kde, stats

warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# One entry per model. 'dir' is an output directory (or a model_*.hdf5 glob); a relative
# path is taken from the SAGE26 top directory. 'color' and 'ls' are optional and override
# the automatic choice below for that model only.
MODELS = [
    {'dir': 'output/millennium/',          'label': 'SAGE26 Millennium'},
    {'dir': 'output/millennium_vanilla/',  'label': 'SAGE26 vanilla'},
]

OUTPUT_DIR = 'output/compare_plots/'    # relative to the SAGE26 top directory
DATA_DIR = None                         # None -> <script_dir>/../data

# Galaxy selection, applied to every model, every snapshot and every plot
PARTICLE_LEN_CUT = 20       # keep galaxies whose halo has Len >= this many particles
MVIR_CUT = 0.0              # keep galaxies with Mvir >= this [Msun] (0 = no cut)
sSFRcut = -11.0             # log10 sSFR [yr^-1] dividing quiescent from star forming

# Model colours: 'colors' takes MODEL_COLORS in order; 'cmap' samples MODEL_CMAP evenly
# across CMAP_RANGE, one colour per model.
COLOR_MODE = 'colors'
MODEL_COLORS = ['black', 'firebrick', 'dodgerblue', 'darkorange', 'forestgreen', 'purple']
MODEL_CMAP = 'viridis'
CMAP_RANGE = (0.0, 0.85)
# Used to tell the models apart on plots whose colours already mean something else
# (redshift bins, gas phases, regimes, ...)
MODEL_LINESTYLES = ['-', '--', ':', '-.']

# Snapshots: None -> each model's latest (local) / full range (history)
LOCAL_SNAP = None
FIRST_SNAP = None
LAST_SNAP = None

whichimf = 1        # 0=Salpeter; 1=Chabrier
dilute = 7500       # Number of galaxies per model to plot in scatter plots

OutputFormat = '.png'
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

# Offset in dex of each IMF's stellar masses above the Chabrier (2003) scale.
# Kroupa (2001) sits 0.04 dex above Chabrier -- the value the Tremonti04, MMAdrews13
# and MMR-Kewley08 headers themselves quote -- and Salpeter 0.24 dex above.
# paper_plots.py carries the same numbers in IMF_TO_CHABRIER_DEX; keep the two in step.
_IMF_ABOVE_CHABRIER_DEX = {
    'chabrier': 0.00,
    'kroupa': 0.04,
    'salpeter': 0.24,
    'diet-salpeter': 0.09,
}


def imf_shift(native):
    """Dex to add to a log10 stellar mass measured with the *native* IMF to put it on the
    IMF selected by whichimf."""
    target = 'salpeter' if whichimf == 0 else 'chabrier'
    return _IMF_ABOVE_CHABRIER_DEX[target] - _IMF_ABOVE_CHABRIER_DEX[native]


# Fields stored in code mass units (1e10 Msun/h); Model.get returns them in Msun
MASS_FIELDS = {
    'CentralMvir', 'Mvir', 'StellarMass', 'MetalsStellarMass', 'BulgeMass', 'BlackHoleMass',
    'ColdGas', 'MetalsColdGas', 'MetalsEjectedMass', 'HotGas', 'MetalsHotGas', 'EjectedMass',
    'CGMgas', 'MetalsCGMgas', 'IntraClusterStars', 'MergerBulgeMass', 'InstabilityBulgeMass',
    'H2gas', 'H1gas', 'infallMvir', 'infallStellarMass', 'MetalsBulgeMass',
    'MetalsIntraClusterStars',
}

SOLAR_MASS_G = 1.989e33     # solar mass [g], as in src/macros.h
CM_PER_KPC = 3.085678e21    # [cm per kpc], CM_PER_MPC / 1e3 in src/macros.h

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))

OutputDir = None    # set in main
DataDir = None      # set in main


def _natural_key(path):
    """Sort model_2.hdf5 before model_10.hdf5 rather than after it."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path)]


def _expand_master(filepath):
    """
    Replace a master index file with the per-core files it links to.

    SAGE also writes a small model.hdf5 next to model_0.hdf5, model_1.hdf5, ... whose
    Core_N groups are external links to those files. It carries no Snap_ groups of its
    own but does claim frac_volume_processed = 1.0, so feeding it in alongside the real
    files would double the summed volume and halve every mass function. Expand it to
    what it points at; drop anything else that holds no snapshots.
    """
    try:
        with h5.File(filepath, 'r') as f:
            if any(key.startswith('Snap_') for key in f.keys()):
                return [filepath]
            base = os.path.dirname(os.path.abspath(filepath))
            linked = []
            for key in f.keys():
                link = f.get(key, getlink=True)
                if isinstance(link, h5.ExternalLink):
                    target = os.path.normpath(os.path.join(base, link.filename))
                    if os.path.isfile(target):
                        linked.append(target)
    except (OSError, KeyError) as exc:
        print(f"  Warning: could not read {filepath} ({exc}); skipping it")
        return []

    if not linked:
        print(f"  Warning: {filepath} contains no snapshots; skipping it")
    return linked


def resolve_input_files(pattern):
    """Turn a directory, file, glob or bare prefix into a sorted, de-duplicated file list."""
    if os.path.isdir(pattern):
        matches = glob.glob(os.path.join(pattern, 'model_*.hdf5'))
        if not matches:  # some runs write a single unnumbered model file
            matches = glob.glob(os.path.join(pattern, '*.hdf5'))
    elif glob.has_magic(pattern):
        matches = glob.glob(pattern)
    elif os.path.isfile(pattern):
        matches = [pattern]
    else:
        matches = glob.glob(pattern + '*.hdf5')

    files = []
    for m in matches:
        if os.path.isfile(m):
            files.extend(_expand_master(m))

    # De-duplicate on the resolved path -- VolumeFraction is summed over the list
    seen = set()
    unique = []
    for f in sorted(files, key=_natural_key):
        key = os.path.realpath(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def read_simulation_params(filepath):
    """Read simulation parameters from an HDF5 file header."""
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['Omega'] = float(sim.attrs['omega_matter'])
        params['OmegaLambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = float(sim.attrs['particle_mass'])

        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])
        params['BaryonFrac'] = float(runtime.attrs['BaryonFrac'])
        params['MShockMsun'] = float(runtime.attrs['MShockMsun'])
        params['UnitMass_Msun'] = float(runtime.attrs['UnitMass_in_g']) / SOLAR_MASS_G
        params['UnitLength_kpc'] = float(runtime.attrs['UnitLength_in_cm']) / CM_PER_KPC

        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])

        snap_numbers = sorted(int(k.replace('Snap_', '')) for k in f.keys() if k.startswith('Snap_'))
        params['available_snapshots'] = snap_numbers
    return params


def model_colors(n):
    """One colour per model, from MODEL_COLORS or sampled from MODEL_CMAP (COLOR_MODE)."""
    if COLOR_MODE == 'cmap':
        if n == 1:
            return [plt.get_cmap(MODEL_CMAP)(CMAP_RANGE[0])]
        return list(plt.get_cmap(MODEL_CMAP)(np.linspace(CMAP_RANGE[0], CMAP_RANGE[1], n)))
    if COLOR_MODE == 'colors':
        if n > len(MODEL_COLORS):
            sys.exit(f"Error: {n} models but only {len(MODEL_COLORS)} MODEL_COLORS; add more "
                     f"or set COLOR_MODE = 'cmap'")
        return MODEL_COLORS[:n]
    sys.exit(f"Error: COLOR_MODE must be 'colors' or 'cmap', not {COLOR_MODE!r}")


class MissingField(Exception):
    """A model's output does not carry a field a plot asked for."""

    def __init__(self, model, field):
        super().__init__(f"{model.label!r} has no field {field!r}")
        self.model = model
        self.field = field


class Model:
    """
    One SAGE26 run: its files, cosmology, style, and cut galaxy properties.

    get(snap, field) returns that field for the galaxies that pass PARTICLE_LEN_CUT and
    MVIR_CUT at that snapshot, with the MASS_FIELDS converted to Msun. Every other field
    comes back in its stored units. Arrays are cached, so asking twice costs nothing.
    A field the output does not carry raises MissingField, and run_plot then leaves this
    model out of that figure.
    """

    def __init__(self, spec, color, ls):
        self.label = spec['label']
        path = spec['dir'] if os.path.isabs(spec['dir']) else os.path.join(TOP_DIR, spec['dir'])
        self.files = resolve_input_files(path)
        if not self.files:
            sys.exit(f"Error: no model files found for {self.label!r} at {path}")
        self.color = spec.get('color', color)
        self.ls = spec.get('ls', ls)

        params = read_simulation_params(self.files[0])
        self.Hubble_h = params['Hubble_h']
        self.BoxSize = params['BoxSize']
        self.Omega = params['Omega']
        self.OmegaLambda = params['OmegaLambda']
        self.PartMass = params['PartMass']
        self.BaryonFrac = params['BaryonFrac']
        self.MShockMsun = params['MShockMsun']
        self.UnitMass_Msun = params['UnitMass_Msun']      # code mass unit in Msun/h
        self.UnitLength_kpc = params['UnitLength_kpc']    # code length unit in kpc/h
        self.redshifts = params['redshifts']
        self.available_snapshots = params['available_snapshots']
        self.VolumeFraction = sum(read_simulation_params(f)['VolumeFraction'] for f in self.files)
        self.volume = (self.BoxSize / self.Hubble_h)**3.0 * self.VolumeFraction

        self.local_snap = LOCAL_SNAP if LOCAL_SNAP is not None else max(self.available_snapshots)
        if self.local_snap not in self.available_snapshots:
            sys.exit(f"Error: snapshot {self.local_snap} not in {self.label!r}")
        first = FIRST_SNAP if FIRST_SNAP is not None else min(self.available_snapshots)
        last = LAST_SNAP if LAST_SNAP is not None else max(self.available_snapshots)
        self.history_snaps = [s for s in self.available_snapshots if first <= s <= last]
        self.FirstSnap = min(self.history_snaps)
        self.LastSnap = max(self.history_snaps)

        with h5.File(self.files[0], 'r') as f:
            self.fields = set(f[f'Snap_{max(self.available_snapshots)}'].keys())

        self._cache = {}
        self._masks = {}
        self._thin_rank = {}

        print(f"  {self.label}: {len(self.files)} file(s), h = {self.Hubble_h}, "
              f"Omega_m = {self.Omega}, Omega_L = {self.OmegaLambda}, f_b = {self.BaryonFrac}, "
              f"box = {self.BoxSize} Mpc/h, volume fraction = {self.VolumeFraction}, "
              f"local snap = {self.local_snap} (z = {self.redshift(self.local_snap):.3f})")

    def redshift(self, snap):
        return self.redshifts[snap]

    def _read_raw(self, snap, field):
        group = f'Snap_{snap}'
        data_list = []
        for filepath in self.files:
            with h5.File(filepath, 'r') as f:
                if group in f and field in f[group]:
                    data = np.array(f[group][field])
                    if data.size > 0:
                        data_list.append(data)
        if not data_list:
            return np.array([])
        return np.concatenate(data_list)

    def mask(self, snap):
        """Boolean mask of the galaxies at snap that pass the particle-length and Mvir cuts."""
        if snap not in self._masks:
            mvir = self._read_raw(snap, 'Mvir') * self.UnitMass_Msun / self.Hubble_h
            keep = mvir >= MVIR_CUT
            if 'Len' in self.fields:
                keep &= self._read_raw(snap, 'Len') >= PARTICLE_LEN_CUT
            self._masks[snap] = keep
        return self._masks[snap]

    def get(self, snap, field):
        if field not in self.fields:
            raise MissingField(self, field)
        key = (snap, field)
        if key not in self._cache:
            data = self._read_raw(snap, field)
            if data.size == 0:
                self._cache[key] = data
            else:
                data = data[self.mask(snap)]
                if field in MASS_FIELDS:
                    data = data * self.UnitMass_Msun / self.Hubble_h
                self._cache[key] = data
        return self._cache[key]

    def thin(self, snap, w, n):
        """
        Pick n of the indices w, the same galaxies however the run is split over files:
        the selection is put in an order fixed by the galaxies' own properties, then drawn
        with a fixed seed.
        """
        w = np.asarray(w)
        if len(w) <= n:
            return w
        if snap not in self._thin_rank:
            keys = [self.get(snap, f) for f in
                    ('ColdGas', 'HotGas', 'BulgeMass', 'BlackHoleMass', 'Mvir', 'StellarMass')]
            rank = np.empty(len(keys[-1]), dtype=np.int64)
            rank[np.lexsort(keys)] = np.arange(len(keys[-1]))
            self._thin_rank[snap] = rank
        w = w[np.argsort(self._thin_rank[snap][w], kind='stable')]
        return np.random.default_rng(2222).choice(w, n, replace=False)


def model_legend_handles(models, lw=2, linestyles=False):
    """Legend entries naming each model: its colour, or its linestyle in black."""
    if linestyles:
        return [Line2D([], [], color='k', ls=m.ls, lw=lw, label=m.label) for m in models]
    return [Line2D([], [], color=m.color, ls='-', lw=lw, label=m.label) for m in models]


def model_panels(models, ncols_max=3, panel_size=(6.5, 5.5), **kwargs):
    """
    Figure with one panel per model, for scatter plots already coloured by a third quantity.
    Returns (fig, axes) with axes a flat list, one per model; unused panels are hidden.
    """
    n = len(models)
    ncols = min(n, ncols_max)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False, **kwargs)
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, m in zip(axes, models):
        ax.set_title(m.label)
    return fig, axes[:n]


def run_plot(plot, models):
    """
    Draw one figure, leaving out any model that lacks a field it needs. A plot that fails
    for any other reason is reported and skipped, so the rest still run.
    """
    models = list(models)
    while models:
        try:
            plot(models)
            return
        except MissingField as exc:
            plt.close('all')
            print(f'  Leaving {exc.model.label!r} out of {plot.__name__}: '
                  f'no {exc.field!r} in its output')
            models = [m for m in models if m is not exc.model]
        except Exception as exc:
            plt.close('all')
            print(f'  Skipping {plot.__name__}: {type(exc).__name__}: {exc}\n')
            return
    print(f'  Skipping {plot.__name__}: no model has the fields it needs\n')


def save_figure(name, fig=None, **kwargs):
    outputFile = OutputDir + name + OutputFormat
    (fig or plt.gcf()).savefig(outputFile, **kwargs)
    print('Saved file to', outputFile, '\n')
    plt.close('all')


def read_obs_data(obs_dir, filename):
    """Read observational data files"""
    filepath = os.path.join(obs_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: Observational data file {filename} not found at {filepath}")
        return None
    return np.loadtxt(filepath)


def find_snapshots_in_z_range(z_min, z_max, redshifts):
    """Find snapshot indices within redshift range"""
    return [i for i, z in enumerate(redshifts) if z_min <= z <= z_max]


# LOCAL PLOTS (single snapshot)
# ==================================================================

# ==================================================================
# Local (single-snapshot) plots, part A
# ==================================================================

def _component_label(models, i, text):
    """Label a coloured component on the first model only; the model legend names the rest."""
    if i > 0:
        return '_nolegend_'
    return f'{models[0].label} {text}' if len(models) == 1 else text


def _add_model_legend(ax, models, loc, linestyles=True):
    """Second legend naming the models, kept alongside the existing one."""
    if len(models) < 2:
        return
    ax.add_artist(ax.get_legend())
    leg = ax.legend(handles=model_legend_handles(models, linestyles=linestyles), loc=loc,
                    frameon=False)
    for t in leg.get_texts():
        t.set_fontsize('medium')


def _style_legend(leg):
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
    return leg


def local_stellar_mass_function(models):
    print('Plotting the stellar mass function, divided by sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

    # Load GAMA morphological SMF data
    # Columns: log_M, E_HE, E_HE_err, cBD, cBD_err, dBD, dBD_err, D, D_err
    gama = np.genfromtxt(os.path.join(DataDir, 'smf', 'gama_smf_morph.ecsv'), comments='#', skip_header=1)
    gama_mass = gama[:, 0]
    gama_E_HE = gama[:, 1]
    gama_D = gama[:, 7]

    # Load Baldry et al. blue/red SMF data
    # Columns: SF_mass, SF_phi, Q_mass, Q_phi (all in log)
    baldry = np.genfromtxt(os.path.join(DataDir, 'morphology', 'baldry_blue_red.csv'), delimiter=',', skip_header=2)
    baldry_sf_mass = baldry[:, 0]
    baldry_sf_phi = baldry[:, 1]
    baldry_q_mass = baldry[:, 2]
    baldry_q_phi = baldry[:, 3]

    for i, m in enumerate(models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        SfrDisk = m.get(snap, 'SfrDisk')
        SfrBulge = m.get(snap, 'SfrBulge')

        w = np.where(StellarMass > 0.0)[0]
        mass = np.log10(StellarMass[w])
        sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )

        mi = np.floor(min(mass)) - 2
        ma = np.floor(max(mass)) + 2
        NB = int((ma - mi) / binwidth)
        (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

        (countsRED, binedges) = np.histogram(mass[sSFR < sSFRcut], range=(mi, ma), bins=NB)
        (countsBLU, binedges) = np.histogram(mass[sSFR > sSFRcut], range=(mi, ma), bins=NB)

        volume = m.volume
        plt.plot(xaxeshisto, np.log10(counts / volume / binwidth), color='k', ls=m.ls, lw=4,
                 label=_component_label(models, i, 'Total'))
        plt.plot(xaxeshisto, np.log10(countsRED / volume / binwidth), color='firebrick', ls=m.ls, lw=4,
                 label=_component_label(models, i, 'Quiescent'))
        plt.plot(xaxeshisto, np.log10(countsBLU / volume / binwidth), color='dodgerblue', ls=m.ls, lw=4,
                 label=_component_label(models, i, 'Star Forming'))

    # Create shaded regions from observations (GAMA + Baldry combined)
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

    plt.fill_between(mass_grid, sf_lower, sf_upper, color='dodgerblue', alpha=0.3, edgecolor='none', label='Observations SF')
    plt.fill_between(mass_grid, q_lower, q_upper, color='firebrick', alpha=0.3, edgecolor='none', label='Observations Q')

    plt.axis([8, 12, -6, -1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.ylabel(r'$\log_{10}\ \phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')

    _style_legend(plt.legend(loc='lower left', numpoints=1, labelspacing=0.1))
    _add_model_legend(ax, models, 'upper right')

    plt.tight_layout()
    save_figure('StellarMassFunction')


def local_baryonic_mass_function(models):
    print('Plotting the baryonic mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

    # Bell et al. 2003 BMF (h=1.0 converted to the first model's h)
    Hubble_h = models[0].Hubble_h
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

    for i, m in enumerate(models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        ColdGas = m.get(snap, 'ColdGas')
        Type = m.get(snap, 'Type')
        volume = m.volume

        w = np.where(StellarMass + ColdGas > 0.0)[0]
        mass = np.log10( (StellarMass[w] + ColdGas[w]) )

        mi = np.floor(min(mass)) - 2
        ma = np.floor(max(mass)) + 2
        NB = int((ma - mi) / binwidth)
        (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

        centrals_mass = mass[Type[w] == 0]
        satellites_mass = mass[Type[w] == 1]

        mi = np.floor(min(centrals_mass)) - 2
        ma = np.floor(max(centrals_mass)) + 2
        NB = int((ma - mi) / binwidth)
        (counts_centrals, binedges_centrals) = np.histogram(centrals_mass, range=(mi, ma), bins=NB)
        xaxeshisto_centrals = binedges_centrals[:-1] + 0.5 * binwidth

        if len(satellites_mass) > 0:
            mi = np.floor(min(satellites_mass)) - 2
            ma = np.floor(max(satellites_mass)) + 2
            NB = int((ma - mi) / binwidth)
            (counts_satellites, binedges_satellites) = np.histogram(satellites_mass, range=(mi, ma), bins=NB)
            xaxeshisto_satellites = binedges_satellites[:-1] + 0.5 * binwidth
        else:
            counts_satellites = np.array([])
            xaxeshisto_satellites = np.array([])

        # Overplot the model histograms
        plt.plot(xaxeshisto, counts / volume / binwidth, color='k', ls=m.ls,
                 label=_component_label(models, i, 'Model') if len(models) > 1 else 'Model')
        plt.plot(xaxeshisto_centrals, counts_centrals / volume / binwidth, color='b', ls=m.ls, lw=2,
                 label=_component_label(models, i, 'Centrals') if len(models) > 1 else 'Model - Centrals')
        if len(counts_satellites) > 0:
            plt.plot(xaxeshisto_satellites, counts_satellites / volume / binwidth, color='g', ls=m.ls, lw=1.5,
                     label=_component_label(models, i, 'Satellites') if len(models) > 1 else 'Model - Satellites')

    plt.yscale('log')
    plt.axis([8.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # and the x-axis labels

    _style_legend(plt.legend(loc='lower left', numpoints=1, labelspacing=0.1))
    _add_model_legend(ax, models, 'upper right')

    plt.tight_layout()
    save_figure('BaryonicMassFunction')


def local_gas_mass_function(models):
    print('Plotting the cold gas mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

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

    Hubble_h = models[0].Hubble_h
    ObrCold_xval = np.log10(10**(ObrCold[:, 0])  /Hubble_h/Hubble_h)
    ObrCold_yval = (10**(ObrCold[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    Zwaan_xval = np.log10(10**(Zwaan[:, 0]) /Hubble_h/Hubble_h)
    Zwaan_yval = (10**(Zwaan[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    ObrRaw_xval = np.log10(10**(ObrRaw[:, 0])  /Hubble_h/Hubble_h)
    ObrRaw_yval = (10**(ObrRaw[:, 1]) * Hubble_h*Hubble_h*Hubble_h)

    plt.plot(ObrCold_xval, ObrCold_yval, color='k', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (Cold Gas)')
    plt.plot(Zwaan_xval, Zwaan_yval, color='cyan', lw = 7, alpha=0.25, label='Zwaan et al. 2005 (HI)')
    plt.plot(ObrRaw_xval, ObrRaw_yval, color='magenta', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (H2)')

    for i, m in enumerate(models):
        snap = m.local_snap
        ColdGas = m.get(snap, 'ColdGas')
        H2gas = m.get(snap, 'H2gas')
        H1gas = m.get(snap, 'H1gas')
        Type = m.get(snap, 'Type')
        volume = m.volume

        w = np.where((ColdGas > 0.0) & (Type==0))[0]
        mass = np.log10(ColdGas[w])
        H2mass = np.log10(H2gas[w])
        H1mass = np.log10(H1gas[w])  # Now read directly from model output

        mi = np.floor(min(mass)) - 2
        ma = np.floor(max(mass)) + 2
        NB = int((ma - mi) / binwidth)

        (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth
        (counts_h2, binedges_h2) = np.histogram(H2mass, range=(mi, ma), bins=NB)
        xaxeshisto_h2 = binedges_h2[:-1] + 0.5 * binwidth
        (counts_h1, binedges_h1) = np.histogram(H1mass, range=(mi, ma), bins=NB)
        xaxeshisto_h1 = binedges_h1[:-1] + 0.5 * binwidth

        multi = len(models) > 1
        plt.plot(xaxeshisto_h2, counts_h2 / volume / binwidth, 'magenta', linestyle=m.ls,
                 label=_component_label(models, i, 'H2 Gas') if multi else 'Model - H2 Gas')
        plt.plot(xaxeshisto_h1, counts_h1 / volume / binwidth, 'cyan', linestyle=m.ls,
                 label=_component_label(models, i, 'HI Gas') if multi else 'Model - HI Gas')
        plt.plot(xaxeshisto, counts / volume / binwidth, color='k', linestyle=m.ls,
                 label=_component_label(models, i, 'Cold Gas') if multi else 'Model - Cold Gas')

    plt.yscale('log')
    plt.axis([8.0, 11.5, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{X}}\ (M_{\odot})$')  # and the x-axis labels

    _style_legend(plt.legend(loc='lower left', numpoints=1, labelspacing=0.1))
    _add_model_legend(ax, models, 'upper right')

    plt.tight_layout()
    save_figure('GasMassFunction')


def local_baryonic_tully_fisher(models):
    print('Plotting the baryonic TF relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        ColdGas = m.get(snap, 'ColdGas')
        BulgeMass = m.get(snap, 'BulgeMass')
        Type = m.get(snap, 'Type')
        Vmax = m.get(snap, 'Vmax')

        w = np.where((Type == 0) & (StellarMass + ColdGas > 0.0) &
          (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
        w = m.thin(snap, w, dilute)

        mass = np.log10( (StellarMass[w] + ColdGas[w]) )
        vel = np.log10(Vmax[w])

        plt.scatter(vel, mass, marker='x', s=1, c=[m.color], alpha=0.9, label=f'{m.label} Sb/c galaxies')

    # overplot Stark, McGaugh & Swatters 2009 (assumes h=0.75? ... what IMF?)
    w = np.arange(0.5, 10.0, 0.5)
    TF = 3.94*w + 1.79
    TF_upper = TF + 0.26
    TF_lower = TF - 0.26

    plt.fill_between(w, TF_lower, TF_upper, color='blue', alpha=0.2)

    plt.ylabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log_{10}V_{max}\ (km/s)$')  # and the x-axis labels

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([1.4, 2.9, 7.5, 12.0])

    _style_legend(plt.legend(loc='lower right', markerscale=6))

    plt.tight_layout()
    save_figure('BaryonicTullyFisher')


def local_specific_star_formation_rate(models):
    print('Plotting the specific sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        SfrDisk = m.get(snap, 'SfrDisk')
        SfrBulge = m.get(snap, 'SfrBulge')

        w = np.where(StellarMass > 0.01)[0]
        w = m.thin(snap, w, dilute)
        mass = np.log10(StellarMass[w])
        sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )
        plt.scatter(mass, sSFR, marker='x', s=1, c=[m.color], alpha=0.9, label=f'{m.label} galaxies')

    # overplot dividing line between SF and passive
    w = np.arange(7.0, 13.0, 1.0)
    plt.plot(w, w/w*sSFRcut, 'b:', lw=2.0)

    plt.ylabel(r'$\log_{10}\ s\mathrm{SFR}\ (\mathrm{yr^{-1}})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([8.0, 12.0, -16.0, -8.0])

    _style_legend(plt.legend(loc='lower right', markerscale=6))

    plt.tight_layout()
    save_figure('SpecificStarFormationRate')


def local_gas_fraction(models):
    print('Plotting the gas fractions')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        ColdGas = m.get(snap, 'ColdGas')
        BulgeMass = m.get(snap, 'BulgeMass')

        w = np.where((StellarMass + ColdGas > 0.0) &
          (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
        w = m.thin(snap, w, dilute)

        mass = np.log10(StellarMass[w])
        fraction = ColdGas[w] / (StellarMass[w] + ColdGas[w])

        plt.scatter(mass, fraction, marker='x', s=1, c=[m.color], alpha=0.9, label=f'{m.label} Sb/c galaxies')

    plt.ylabel(r'$\mathrm{Cold\ Mass\ /\ (Cold+Stellar\ Mass)}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([8.0, 12.0, 0.0, 1.0])

    _style_legend(plt.legend(loc='upper right', markerscale=6))

    plt.tight_layout()
    save_figure('GasFraction')


def local_metallicity(models):
    print('Plotting the metallicities')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        ColdGas = m.get(snap, 'ColdGas')
        MetalsColdGas = m.get(snap, 'MetalsColdGas')
        Type = m.get(snap, 'Type')

        w = np.where((Type == 0) & (ColdGas / (StellarMass + ColdGas) > 0.1) & (StellarMass > 1.0e8))[0]
        w = m.thin(snap, w, dilute)

        mass = np.log10(StellarMass[w])
        Z = np.log10((MetalsColdGas[w]) / ColdGas[w] / 0.02) + 9.0  # Convert to 12 + log(O/H) scale, assuming solar metallicity of 0.02 and that all metals are oxygen (for simplicity)
        plt.scatter(mass, Z, marker='x', s=1, c=[m.color], alpha=0.9, label=f'{m.label} galaxies')

    # Tremonti et al. 2004 - the primary observational reference
    try:
        tremonti_data = np.loadtxt(os.path.join(DataDir, 'metallicity', 'Tremonti04.dat'))
        tremonti_mass = tremonti_data[:, 0]
        tremonti_Z = tremonti_data[:, 1]
        tremonti_Z_err_low = tremonti_data[:, 2]
        tremonti_Z_err_high = tremonti_data[:, 3]
        # Tremonti+04 masses are Kroupa, per the file header.
        tremonti_mass_corrected = tremonti_mass + imf_shift('kroupa')
        ax.plot(tremonti_mass_corrected, tremonti_Z, '-', color='red', linewidth=1.5, alpha=0.7, label='Tremonti+04')
        ax.fill_between(tremonti_mass_corrected, tremonti_Z_err_low, tremonti_Z_err_high, color='firebrick', alpha=0.2, zorder=5)
    except Exception as e:
        print(f"Warning: Could not load Tremonti04.dat: {e}. Using fallback polynomial fit.")
        w_obs = np.arange(7.0, 13.0, 0.1)
        Zobs = -1.492 + 1.847*w_obs - 0.08026*w_obs*w_obs
        ax.plot(w_obs + imf_shift('kroupa'), Zobs, 'o-', linewidth=3,
                label='Tremonti et al. 2004 (poly fit)')

    # Curti et al. 2020
    try:
        curti_data = np.loadtxt(os.path.join(DataDir, 'metallicity', 'Curti2020.dat'))
        # Kroupa: Curti+20 takes its masses from the MPA-JHU catalogue, as
        # Tremonti+04 and Andrews & Martini+13 do.
        curti_mass = curti_data[:, 0] + imf_shift('kroupa')
        curti_Z = curti_data[:, 1]
        curti_Z_low = curti_data[:, 2]
        curti_Z_high = curti_data[:, 3]
        ax.plot(curti_mass, curti_Z, linestyle='-', color='blue', linewidth=2, label='Curti+20')
        ax.fill_between(curti_mass, curti_Z_low, curti_Z_high, color='darkblue', alpha=0.2, zorder=5)
    except Exception as e:
        print(f"Warning: Could not load Curti2020.dat: {e}")

    # Andrews & Martini 2013
    try:
        andrews_data = np.loadtxt(os.path.join(DataDir, 'metallicity', 'MMAdrews13.dat'))
        andrews_mass = andrews_data[:, 0]
        andrews_Z = andrews_data[:, 1]
        # Kroupa, per the file header.
        andrews_mass_corrected = andrews_mass + imf_shift('kroupa')
        ax.scatter(andrews_mass_corrected, andrews_Z, marker='s', s=30, color='green', edgecolors='darkgreen', linewidth=0.5, alpha=0.8, label='Andrews & Martini 2013')
    except Exception as e:
        print(f"Warning: Could not load MMAdrews13.dat: {e}")

    # Kewley & Ellison 2008 - T04 calibration (most commonly used)
    try:
        kewley_data = np.loadtxt(os.path.join(DataDir, 'metallicity', 'MMR-Kewley08.dat'))
        t04_start = 59
        t04_end = 74
        kewley_mass_t04 = kewley_data[t04_start:t04_end, 0]
        kewley_Z_t04 = kewley_data[t04_start:t04_end, 1]
        # Kroupa, per the file header.
        kewley_mass_corrected = kewley_mass_t04 + imf_shift('kroupa')
        ax.scatter(kewley_mass_corrected, kewley_Z_t04, marker='D', s=40, color='yellow', edgecolors='goldenrod', linewidth=0.8, alpha=0.8, label='Kewley & Ellison 2008')
    except Exception as e:
        print(f"Warning: Could not load MMR-Kewley08.dat: {e}")

    # Gallazzi et al. 2005 - Stellar metallicity (note: this is different from gas metallicity)
    try:
        gallazzi_data = np.loadtxt(os.path.join(DataDir, 'metallicity', 'MSZR-Gallazzi05.dat'))
        gallazzi_mass = gallazzi_data[7:, 0]
        gallazzi_Z_stellar = gallazzi_data[7:, 1]
        gallazzi_Z_gas_approx = gallazzi_Z_stellar + 8.69
        # Chabrier, per the file header.
        gallazzi_mass_corrected = gallazzi_mass + imf_shift('chabrier')
        ax.scatter(gallazzi_mass_corrected, gallazzi_Z_gas_approx, marker='P', s=100, color='k', edgecolors='gray', linewidth=0.5, alpha=0.8, label='Gallazzi+05')
    except Exception as e:
        print(f"Warning: Could not load MSZR-Gallazzi05.dat: {e}")

    plt.ylabel(r'$12\ +\ \log_{10}(\mathrm{O/H})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')  # and the x-axis labels

    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.axis([8.0, 12.0, 8.0, 9.5])

    leg = _style_legend(plt.legend(loc='lower right'))
    for handle in leg.legend_handles[:len(models)]:  # model points are too small to see otherwise
        handle.set_sizes([30])

    plt.tight_layout()
    save_figure('Metallicity')


def local_black_hole_bulge_relationship(models):
    print('Plotting the black hole-bulge relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    for m in models:
        snap = m.local_snap
        BulgeMass = m.get(snap, 'BulgeMass')
        BlackHoleMass = m.get(snap, 'BlackHoleMass')

        w = np.where((BulgeMass > 1.0e8) & (BlackHoleMass > 01.0e6))[0]
        w = m.thin(snap, w, dilute)

        bh = np.log10(BlackHoleMass[w])
        bulge = np.log10(BulgeMass[w])

        plt.scatter(bulge, bh, marker='x', s=1, c=[m.color], alpha=0.9, label=f'{m.label} galaxies', zorder=10)

        # how many black holes are there in this plot?
        print(f"{m.label}: number of model black holes plotted: {len(BlackHoleMass)}")
        print(f"{m.label}: number of model black holes with mass > 0: {np.sum(BlackHoleMass > 0)}")

    # overplot Haring & Rix 2004
    w = 10. ** np.arange(20)
    BHdata = 10. ** (8.2 + 1.12 * np.log10(w / 1.0e11))
    plt.plot(np.log10(w), np.log10(BHdata), 'b-', label=r"Haring \& Rix 2004")

    # Observational points (h scaling from the first model)
    Hubble_h = models[0].Hubble_h
    M_BH_obs = (0.7/Hubble_h)**2*1e8*np.array([39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39, 0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12, 1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4, 0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5, 0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06, 0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10, 3.7, 0.55, 13, 0.11])
    M_BH_hi = (0.7/Hubble_h)**2*1e8*np.array([4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26, 0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005, 0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3, 0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3, 0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2, 0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004, 0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = (0.7/Hubble_h)**2*1e8*np.array([5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09, 0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005, 0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3, 0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5, 0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1, 0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004, 0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = (0.7/Hubble_h)**2*1e10*np.array([69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9, 4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88, 1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6, 3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9, 1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29, 11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = (0.7/Hubble_h)**2*1e10*np.array([59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7, 6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28, 2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8, 2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2, 6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10, 17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = (0.7/Hubble_h)**2*1e10*np.array([32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1, 2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52, 1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5, 1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6, 3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9, 1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,1,0])
    yerr2, yerr1 = np.log10((M_BH_obs+M_BH_hi)/M_BH_obs), -np.log10((M_BH_obs-M_BH_lo)/M_BH_obs)
    xerr2, xerr1 = np.log10((M_sph_obs+M_sph_hi)/M_sph_obs), -np.log10((M_sph_obs-M_sph_lo)/M_sph_obs)
    plt.errorbar(np.log10(M_sph_obs[core==0]), np.log10(M_BH_obs[core==0]), yerr=[yerr1[core==0],yerr2[core==0]], xerr=[xerr1[core==0],xerr2[core==0]], color='orange', alpha=0.6, label=r'S13 core', ls='none', lw=2, ms=0)
    plt.errorbar(np.log10(M_sph_obs[core==1]), np.log10(M_BH_obs[core==1]), yerr=[yerr1[core==1],yerr2[core==1]], xerr=[xerr1[core==1],xerr2[core==1]], color='c', alpha=0.6, label=r'S13 Sersic', ls='none', lw=2, ms=0)

    plt.ylabel(r'$\log\ M_{\mathrm{BH}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log\ M_{\mathrm{bulge}}\ (M_{\odot})$')  # and the x-axis labels

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([8.0, 12.0, 6.0, 10.0])

    _style_legend(plt.legend(loc='upper left', markerscale=6))

    plt.tight_layout()
    save_figure('BlackHoleBulgeRelationship')


def local_quiescent_fraction(models):
    print('Plotting the quiescent fraction vs stellar mass')

    # One panel per model: the satellite lines already use linestyle for halo mass
    fig, axes = model_panels(models, sharex=True, sharey=True)

    groupscale = 12.5

    for ax, m in zip(axes, models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        CentralMvir = m.get(snap, 'CentralMvir')
        Type = m.get(snap, 'Type')
        SfrDisk = m.get(snap, 'SfrDisk')
        SfrBulge = m.get(snap, 'SfrBulge')

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
        ax.plot(Mass[w], Fraction[w], label='All')

        w = np.where(CentralFraction > 0)[0]
        ax.plot(Mass[w], CentralFraction[w], color='Blue', label='Centrals')

        w = np.where(SatelliteFraction > 0)[0]
        ax.plot(Mass[w], SatelliteFraction[w], color='Red', label='Satellites')

        w = np.where(SatelliteFractionLo > 0)[0]
        ax.plot(Mass[w], SatelliteFractionLo[w], 'r--', label='Satellites-Lo')

        w = np.where(SatelliteFractionHi > 0)[0]
        ax.plot(Mass[w], SatelliteFractionHi[w], 'r-.', label='Satellites-Hi')

        ax.set_xlabel(r'$\log_{10} M_{\mathrm{stellar}}\ (M_{\odot})$')  # Set the x-axis label

        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

        ax.axis([9.5, 12.5, 0.0, 1.05])

    axes[0].set_ylabel(r'$\mathrm{Quescient\ Fraction}$')  # Set the y-axis label
    _style_legend(axes[0].legend(loc='lower right'))

    plt.tight_layout()
    save_figure('QuiescentFraction', fig)


def local_bulge_mass_fraction(models):
    print('Plotting the mass fraction of galaxies')

    plt.figure()
    ax = plt.subplot(111)

    binwidth = 0.5
    shift = binwidth/2.0
    mass_range = np.arange(8.5-shift, 12.5+shift, binwidth)
    bins = len(mass_range)

    for k, m in enumerate(models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        BulgeMass = m.get(snap, 'BulgeMass')

        w = np.where(StellarMass > 0.0)[0]
        fBulge = BulgeMass[w] / StellarMass[w]
        fDisk = 1.0 - fBulge
        mass = np.log10(StellarMass[w])

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

        multi = len(models) > 1
        w = np.where(fBulge_ave > 0.0)[0]
        plt.plot(mass_range[w]+shift, fBulge_ave[w], color='r', ls=m.ls,
                 label=_component_label(models, k, 'bulge') if multi else 'bulge')
        plt.fill_between(mass_range[w]+shift,
            fBulge_ave[w]+fBulge_var[w],
            fBulge_ave[w]-fBulge_var[w],
            facecolor='red', alpha=0.25)

        w = np.where(fDisk_ave > 0.0)[0]
        plt.plot(mass_range[w]+shift, fDisk_ave[w], color='k', ls=m.ls,
                 label=_component_label(models, k, 'disk stars') if multi else 'disk stars')
        plt.fill_between(mass_range[w]+shift,
            fDisk_ave[w]+fDisk_var[w],
            fDisk_ave[w]-fDisk_var[w],
            facecolor='k', alpha=0.25)

    plt.axis([mass_range[0], mass_range[bins-1], 0.0, 1.05])

    plt.ylabel(r'$\mathrm{Stellar\ Mass\ Fraction}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    _style_legend(plt.legend(loc='upper right', numpoints=1, labelspacing=0.1))
    _add_model_legend(ax, models, 'center right')

    plt.tight_layout()
    save_figure('BulgeMassFraction')


def local_bulge_to_total(models):
    print('Plotting the bulge-to-total ratio')

    # One panel per model: the median, merger and instability lines already use linestyle
    fig, axes = model_panels(models, sharex=True, sharey=True)

    moffett = read_obs_data(os.path.join(DataDir, 'morphology'), 'Moffet16.dat')

    binwidth = 0.25
    bin_edges = np.arange(8.5, 12.5 + binwidth, binwidth)
    bin_centres = bin_edges[:-1] + binwidth / 2.0

    for ax, m in zip(axes, models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        BulgeMass = m.get(snap, 'BulgeMass')
        MergerBulgeMass = m.get(snap, 'MergerBulgeMass')
        InstabilityBulgeMass = m.get(snap, 'InstabilityBulgeMass')

        w = np.where((StellarMass > 0.0) & (BulgeMass > 0.0))[0]
        mass = np.log10(StellarMass[w])
        BtoT = BulgeMass[w] / StellarMass[w]
        BtoT_merger = MergerBulgeMass[w] / StellarMass[w]
        BtoT_instab = InstabilityBulgeMass[w] / StellarMass[w]

        digitized = np.digitize(mass, bin_edges)

        # Median B/T with the 16-84 percentile spread, plus the mean split into the
        # merger-driven and disc-instability-driven bulge components. The two
        # components are means rather than medians so that they add to the mean B/T.
        BtoT_med = np.full(len(bin_centres), np.nan)
        BtoT_lo = np.full(len(bin_centres), np.nan)
        BtoT_hi = np.full(len(bin_centres), np.nan)
        merger_ave = np.full(len(bin_centres), np.nan)
        instab_ave = np.full(len(bin_centres), np.nan)

        for i in range(len(bin_centres)):
            sel = digitized == i + 1
            if np.sum(sel) < 5:  # too few galaxies for a meaningful percentile
                continue
            BtoT_med[i] = np.median(BtoT[sel])
            BtoT_lo[i], BtoT_hi[i] = np.percentile(BtoT[sel], [16.0, 84.0])
            merger_ave[i] = np.mean(BtoT_merger[sel])
            instab_ave[i] = np.mean(BtoT_instab[sel])

        good = np.isfinite(BtoT_med)
        ax.fill_between(bin_centres[good], BtoT_lo[good], BtoT_hi[good],
            facecolor='k', alpha=0.2, label='Model 16-84th percentile')
        ax.plot(bin_centres[good], BtoT_med[good], 'k-', lw=2, label='Model median')
        ax.plot(bin_centres[good], merger_ave[good], '--', color='firebrick', lw=1.5,
            label='Mean merger-driven')
        ax.plot(bin_centres[good], instab_ave[good], '-.', color='royalblue', lw=1.5,
            label='Mean instability-driven')

        ax.scatter(mass, BtoT, color='k', s=0.5, zorder=10, alpha=0.45)  # points on top of lines

        # Moffett et al. (2016) GAMA bulge mass fractions
        if moffett is not None:
            obs_mass = moffett[:, 0] + imf_shift('chabrier')
            ax.errorbar(obs_mass, moffett[:, 1],
                yerr=[moffett[:, 1] - moffett[:, 2], moffett[:, 3] - moffett[:, 1]],
                fmt='o', ms=4, color='darkorange', mfc='darkorange', ecolor='darkorange',
                elinewidth=1, capsize=2, zorder=5, label='Moffett et al. (2016)')

        ax.axis([8.5, 12.5, 0.0, 1.05])
        ax.set_xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    axes[0].set_ylabel(r'$M_{\mathrm{bulge}} / M_{\mathrm{stars}}$')  # Set the y...
    _style_legend(axes[0].legend(loc='upper left', numpoints=1, labelspacing=0.1))

    plt.tight_layout()
    save_figure('BulgeToTotal', fig)


LOCAL_PLOTS_A = [
    local_stellar_mass_function,
    local_baryonic_mass_function,
    local_gas_mass_function,
    local_baryonic_tully_fisher,
    local_specific_star_formation_rate,
    local_gas_fraction,
    local_metallicity,
    local_black_hole_bulge_relationship,
    local_quiescent_fraction,
    local_bulge_mass_fraction,
    local_bulge_to_total,
]


# ------------------------------------------------------- local plots, part B

def _model_legend(ax, models, loc='best', **kwargs):
    """Add a second legend naming the models by linestyle, when there is more than one."""
    if len(models) > 1:
        first = ax.get_legend()
        leg = ax.legend(handles=model_legend_handles(models, linestyles=True), loc=loc,
                        frameon=False, **kwargs)
        if first is not None:
            ax.add_artist(first)
        return leg


def local_baryon_fraction(models):
    print('Plotting the average baryon fraction vs halo mass (can take time)')

    plt.figure()
    ax = plt.subplot(111)

    MinHalo, MaxHalo, Interval = 11.0, 16.0, 0.1
    HaloBins = np.arange(MinHalo, MaxHalo + Interval, Interval)
    Nbins = len(HaloBins) - 1

    # (name, label, colour, linestyle in the single-model plot)
    components = [('Baryons', 'Total', 'k', '-'),
                  ('StellarMass', 'Stars', 'magenta', '--'),
                  ('ColdGas', 'Cold gas', 'blue', ':'),
                  ('HotGas', 'Hot gas', 'red', '-'),
                  ('CGMgas', 'Circumgalactic Medium', 'green', '-.'),
                  ('IntraClusterStars', 'Intracluster Stars', 'orange', '-.'),
                  ('BlackHoleMass', None, None, None),
                  ('EjectedMass', 'Ejected gas', 'yellow', '--')]

    for baryon_frac in sorted({m.BaryonFrac for m in models}):
        plt.axhline(y=baryon_frac, color='grey', linestyle='--', linewidth=1.0,
                    label='Baryon Fraction = {:.2f}'.format(baryon_frac))

    for k, m in enumerate(models):
        baryon_frac = m.BaryonFrac
        snap = m.local_snap
        Mvir = m.get(snap, 'Mvir')
        Type = m.get(snap, 'Type')
        CentralGalaxyIndex = m.get(snap, 'CentralGalaxyIndex')
        masses = {name: m.get(snap, name) for name, _, _, _ in components[1:]}
        masses['Baryons'] = sum(masses[name] for name, _, _, _ in components[1:])

        with np.errstate(divide='ignore'):
            HaloMass = np.log10(Mvir)
        bin_indices = np.digitize(HaloMass, HaloBins) - 1

        # Sum every reservoir over all galaxies sharing a CentralGalaxyIndex
        keys, inverse = np.unique(CentralGalaxyIndex, return_inverse=True)
        halo_sums = {name: np.bincount(inverse, weights=arr, minlength=len(keys))
                     for name, arr in masses.items()}

        MeanCentralHaloMass = []
        means = {name: [] for name in masses}
        errs = {name: [] for name in masses}
        for i in range(Nbins - 1):
            w1 = np.where((Type == 0) & (bin_indices == i))[0]
            if len(w1) > 10:
                MeanCentralHaloMass.append(np.mean(np.log10(Mvir[w1])))
                sqrt_n = np.sqrt(len(w1))
                for name in masses:
                    frac = halo_sums[name][inverse[w1]] / Mvir[w1]
                    means[name].append(np.mean(frac))
                    errs[name].append(np.std(frac) / sqrt_n)

        MeanCentralHaloMass = np.array(MeanCentralHaloMass)
        for name in masses:
            means[name] = np.array(means[name])
            errs[name] = np.array(errs[name])

        print('  {}: total baryon fraction per halo mass bin (cosmic f_b = {:.3f}):'.format(m.label, baryon_frac))
        over_bins = 0
        for hm, bf in zip(MeanCentralHaloMass, means['Baryons']):
            flag = '  <-- ABOVE cosmic f_b' if bf > baryon_frac else ''
            if bf > baryon_frac:
                over_bins += 1
            print('    log10(Mvir) = {:5.2f}   f_b = {:.4f}{}'.format(hm, bf, flag))
        if over_bins > 0:
            print('  WARNING: {} bin(s) exceed cosmic baryon fraction.'.format(over_bins))
        else:
            print('  All bins are below the cosmic baryon fraction.')

        for name, label, color, ls in components:
            if label is None:
                continue
            lower = np.maximum(means[name] - errs[name], 1e-6)  # Prevent negative values on log scale
            plt.fill_between(MeanCentralHaloMass, lower, means[name] + errs[name], color=color, alpha=0.2)
            plt.plot(MeanCentralHaloMass, means[name], color=color,
                     linestyle=ls if len(models) == 1 else m.ls,
                     label=label if k == 0 else None)

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} \mathrm{Baryon\ Fraction}$')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.axis([11.1, 15.0, 0.0, 0.2])

    leg = plt.legend(loc='upper right', numpoints=1, labelspacing=0.1, bbox_to_anchor=(1.0, 0.5))
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')
    _model_legend(ax, models, loc='upper left')

    plt.tight_layout()
    save_figure('BaryonFraction')


def local_mass_reservoir_scatter(models):
    print('Plotting the mass in stellar, cold, hot, ejected, ICS reservoirs')

    fig, axes = model_panels(models, sharex=True, sharey=True)
    dilute_mass_reservoir = 10000

    for ax, m in zip(axes, models):
        snap = m.local_snap
        Mvir = m.get(snap, 'Mvir')
        StellarMass = m.get(snap, 'StellarMass')
        # Halo mass minimum is MVIR_CUT, applied by the Model
        w = np.where((m.get(snap, 'Type') == 0) & (Mvir > 0.0) & (StellarMass > 0.0))[0]
        if(len(w) > dilute_mass_reservoir): w = m.thin(snap, w, dilute_mass_reservoir)

        HaloMass = np.log10(Mvir[w])
        with np.errstate(divide='ignore'):
            ax.scatter(HaloMass, np.log10(StellarMass[w]), marker='x', s=0.3, c='w', alpha=0.5, label='Stars')
            ax.scatter(HaloMass, np.log10(m.get(snap, 'ColdGas')[w]), marker='x', s=0.3, color='blue', alpha=0.5, label='Cold gas')
            ax.scatter(HaloMass, np.log10(m.get(snap, 'HotGas')[w]), marker='x', s=0.3, color='red', alpha=0.5, label='Hot gas')
            ax.scatter(HaloMass, np.log10(m.get(snap, 'EjectedMass')[w]), marker='x', s=0.3, color='green', alpha=0.5, label='Ejected gas')
            ax.scatter(HaloMass, np.log10(m.get(snap, 'IntraClusterStars')[w]), marker='x', s=0.3, color='yellow', alpha=0.5, label='Intracluster stars')
            ax.scatter(HaloMass, np.log10(m.get(snap, 'CGMgas')[w]), marker='x', s=0.3, color='orange', alpha=0.5, label='CGM gas')

        ax.set_xlabel(r'$\log\ M_{\mathrm{vir}}\ (h^{-1}\ M_{\odot})$')
        ax.axis([10.0, 15.0, 7.5, 14.0])

    axes[0].set_ylabel(r'$\mathrm{stellar,\ cold,\ hot,\ ejected,\ CGM,\ ICS\ mass}$')
    leg = axes[0].legend(loc='upper left', markerscale=10)
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    plt.tight_layout()
    save_figure('MassReservoirScatter', fig)


def local_spatial_distribution(models):
    print('Plotting the spatial distribution of all galaxies')

    fig = plt.figure(figsize=(18, 5 * len(models)))

    for k, m in enumerate(models):
        snap = m.local_snap
        BoxSize = m.BoxSize
        w = np.where((m.get(snap, 'Mvir') > 0.0) & (m.get(snap, 'StellarMass') > 1.0e9))[0]
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        xx = m.get(snap, 'Posx')[w]
        yy = m.get(snap, 'Posy')[w]
        zz = m.get(snap, 'Posz')[w]

        buff = BoxSize*0.1

        for j, (a, b, ylab, xlab) in enumerate([(xx, yy, 'x', 'y'), (xx, zz, 'x', 'z'), (yy, zz, 'y', 'z')]):
            ax = fig.add_subplot(len(models), 3, 3 * k + j + 1)
            ab = np.vstack([a, b])
            density = gaussian_kde(ab)(ab)
            ax.scatter(a, b, marker='o', s=0.3, c=density, cmap='plasma', alpha=0.5)
            ax.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
            ax.set_ylabel(r'$\mathrm{' + ylab + '}$')
            ax.set_xlabel(r'$\mathrm{' + xlab + '}$')
            if j == 1:
                ax.set_title(m.label)

    # Set face color to black for all 2D plots
    for ax in fig.axes:
        ax.set_facecolor('black')
        ax.grid(False)
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

    plt.tight_layout()
    save_figure('SpatialDistribution', fig)


def local_spatial_distribution_3d_box(models):
    print('Plotting the spatial distribution of all galaxies in 3D Box')

    fig, axes = model_panels(models, subplot_kw={'projection': '3d'})

    for ax, m in zip(axes, models):
        snap = m.local_snap
        BoxSize = m.BoxSize
        w = np.where((m.get(snap, 'Mvir') > 0.0) & (m.get(snap, 'StellarMass') > 1.0e9))[0]
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        xx = m.get(snap, 'Posx')[w]
        yy = m.get(snap, 'Posy')[w]
        zz = m.get(snap, 'Posz')[w]

        xyz = np.vstack([xx,yy,zz])
        density = gaussian_kde(xyz)(xyz)
        ax.scatter(xx, yy, zz, s=0.3, c=density, cmap='plasma', alpha=0.8)

        # Draw the box edges
        points = np.array([[0,0,0], [BoxSize,0,0], [BoxSize,BoxSize,0], [0,BoxSize,0],
                           [0,0,BoxSize], [BoxSize,0,BoxSize], [BoxSize,BoxSize,BoxSize], [0,BoxSize,BoxSize]])
        edges = [[points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],
                 [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],
                 [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]]
        ax.add_collection3d(Line3DCollection(edges, colors='k'))

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
    save_figure('SpatialDistribution3D_Box', fig)


def local_star_formation_rate(models):
    print('Plotting the SFR')

    plt.figure()
    ax = plt.subplot(111)
    dilute_sfr = 10000

    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        w2 = np.where(StellarMass > 0.01)[0]
        if(len(w2) > dilute_sfr): w2 = m.thin(snap, w2, dilute_sfr)
        mass = np.log10(StellarMass[w2])
        starformationrate = (m.get(snap, 'SfrDisk')[w2] + m.get(snap, 'SfrBulge')[w2])

        with np.errstate(divide='ignore'):
            plt.scatter(mass, np.log10(starformationrate), c=[m.color], marker='x', s=1, alpha=0.9, label=m.label)

    plt.ylabel(r'$\log_{10} \mathrm{SFR}\ (M_{\odot}\ \mathrm{yr^{-1}})$')
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.xlim(6.0, 12.2)
    plt.ylim(-5, 3)

    if len(models) > 1:
        plt.legend(loc='upper left', frameon=False, markerscale=10)

    plt.tight_layout()
    save_figure('StarFormationRate')


def local_mass_loading_vs_vvir(models):
    print('Plotting mass loading factor vs Vvir')

    plt.figure()
    plotted = False
    for m in models:
        snap = m.local_snap
        MassLoading = m.get(snap, 'MassLoading')
        w = np.where((m.get(snap, 'StellarMass') > 0.0) & (MassLoading > 0.0))[0]
        if len(w) == 0:
            print(f'  {m.label}: no galaxies with MassLoading > 0 (FIRE mass-loading mode off).')
            continue
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        plt.scatter(np.log10(m.get(snap, 'Vvir')[w]), MassLoading[w], c=[m.color], marker='x', s=1,
                    alpha=0.9, label=m.label)
        plotted = True

    if not plotted:
        print('  Skipping mass_loading_vs_vvir: no galaxies with MassLoading > 0 '
              '(FIRE mass-loading mode off).\n')
        plt.close()
        return

    plt.xlabel(r'$\log_{10} V_{\mathrm{vir}}\ (\mathrm{km/s})$')
    plt.ylabel(r'$\eta_{\mathrm{reheat}} = \dot{M}_{\mathrm{reheat}} / \dot{M}_{\star}$')
    if len(models) > 1:
        plt.legend(loc='upper right', frameon=False, markerscale=10)

    plt.tight_layout()
    save_figure('mass_loading_vs_vvir')


def local_stellar_vs_halo_mass_by_regime(models):
    print('Plotting stellar mass vs halo mass colored by regime')

    fig, axes = model_panels(models, sharex=True, sharey=True)

    for ax, m in zip(axes, models):
        snap = m.local_snap
        Mvir = m.get(snap, 'Mvir')
        StellarMass = m.get(snap, 'StellarMass')
        w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        log10_halo_mass = np.log10(Mvir[w])
        log10_stellar_mass = np.log10(StellarMass[w])
        regime_values = m.get(snap, 'Regime')[w]

        cgm_regime = (regime_values == 0)
        hot_regime = (regime_values == 1)

        if np.any(cgm_regime):
            ax.scatter(log10_halo_mass[cgm_regime], log10_stellar_mass[cgm_regime],
                       c='blue', s=1, alpha=0.6, label='CGM Regime')
        if np.any(hot_regime):
            ax.scatter(log10_halo_mass[hot_regime], log10_stellar_mass[hot_regime],
                       c='red', s=1, alpha=0.6, label='Hot Regime')

        ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
        ax.set_xlim(10, 15)
        ax.set_ylim(8, 12)
        ax.legend(loc='upper left', frameon=False, markerscale=5)

    axes[0].set_ylabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.tight_layout()
    save_figure('stellar_vs_halo_mass_by_regime', fig)


def local_plot_density_contours(ax, x, y, x_bins, y_bins, color, label, clip_above=None, clip_below=None):
    """Plot filled contours with 1, 2, 3 sigma levels"""
    from scipy.ndimage import gaussian_filter

    if len(x) < 10:
        return

    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    H = H.T  # Transpose for correct orientation

    H_smooth = gaussian_filter(H, sigma=1.5)

    y_centers = (yedges[:-1] + yedges[1:]) / 2
    if clip_above is not None:
        mask_2d = y_centers[:, np.newaxis] <= clip_above
        H_smooth = H_smooth * mask_2d
    if clip_below is not None:
        mask_2d = y_centers[:, np.newaxis] >= clip_below
        H_smooth = H_smooth * mask_2d

    sorted_H = np.sort(H_smooth.flatten())[::-1]
    sorted_H = sorted_H[sorted_H > 0]
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

    for i, (level, alpha) in enumerate(zip(levels, alphas)):
        if i == len(levels) - 1:
            ax.contourf(x_centers, y_centers, H_smooth,
                        levels=[level, H_smooth.max()],
                        colors=[color], alpha=alpha, label=label)
        else:
            ax.contourf(x_centers, y_centers, H_smooth,
                        levels=[level, levels[i+1] if i+1 < len(levels) else H_smooth.max()],
                        colors=[color], alpha=alpha)

    ax.contour(x_centers, y_centers, H_smooth,
               levels=levels, colors=color, linewidths=1.0, alpha=0.8)


def local_ssfr_vs_stellar_mass(models):
    print('Plotting specific SFR vs stellar mass')

    fig, axes = model_panels(models, panel_size=(10, 8), sharex=True, sharey=True)
    dilute_ssfr = 100000

    x_bins = np.linspace(8.0, 12.2, 100)
    y_bins = np.linspace(-13, -8, 100)

    for ax, m in zip(axes, models):
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        w2 = np.where(StellarMass > 0.0)[0]
        if(len(w2) > dilute_ssfr): w2 = m.thin(snap, w2, dilute_ssfr)
        mass = np.log10(StellarMass[w2])
        starformationrate = (m.get(snap, 'SfrDisk')[w2] + m.get(snap, 'SfrBulge')[w2])
        sSFR = np.full_like(starformationrate, -99.0)
        mask = (StellarMass[w2] > 0) & (starformationrate > 0)
        sSFR[mask] = np.log10(starformationrate[mask] / StellarMass[w2][mask])

        sf_mask = (sSFR > sSFRcut) & (sSFR > -99.0)  # Star-forming
        q_mask = (sSFR <= sSFRcut) & (sSFR > -99.0)  # Quiescent

        if np.any(q_mask):
            local_plot_density_contours(ax, mass[q_mask], sSFR[q_mask], x_bins, y_bins,
                                        'firebrick', 'Quiescent', clip_above=sSFRcut)
        if np.any(sf_mask):
            local_plot_density_contours(ax, mass[sf_mask], sSFR[sf_mask], x_bins, y_bins,
                                        'dodgerblue', 'Star-forming', clip_below=sSFRcut)
        ax.axhline(y=sSFRcut, color='k', linestyle='--', linewidth=2,
                   label=f'sSFR cut = {sSFRcut}', zorder=10)

        ax.set_xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$', fontsize=14)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax.set_xlim(8.0, 12.2)
        ax.set_ylim(-13, -8)
        ax.legend(loc='upper right', fontsize=12, frameon=False)

    axes[0].set_ylabel(r'$\log_{10} \mathrm{sSFR}\ (\mathrm{yr^{-1}})$', fontsize=14)
    plt.tight_layout()
    save_figure('specific_star_formation_rate', fig, dpi=150)


def local_concentration_vs_mvir(models):
    print('Plotting concentration vs Mvir colored by stellar mass (centrals & satellites)')

    fig, axes = plt.subplots(len(models), 2, figsize=(16, 7 * len(models)), sharey=True, squeeze=False)

    # Shared color limits across all models
    bases = []
    for m in models:
        snap = m.local_snap
        StellarMass = m.get(snap, 'StellarMass')
        bases.append(np.where((m.get(snap, 'Concentration') > 0) & (m.get(snap, 'Mvir') > 0)
                              & (StellarMass > 1.0e8))[0])
    vmin = min(np.log10(m.get(m.local_snap, 'StellarMass')[b]).min() for m, b in zip(models, bases) if len(b))
    vmax = max(np.log10(m.get(m.local_snap, 'StellarMass')[b]).max() for m, b in zip(models, bases) if len(b))

    for row, m, base in zip(axes, models, bases):
        snap = m.local_snap
        Type = m.get(snap, 'Type')
        Mvir = m.get(snap, 'Mvir')
        StellarMass = m.get(snap, 'StellarMass')
        Concentration = m.get(snap, 'Concentration')
        w_cen = base[Type[base] == 0]
        w_sat = base[Type[base] > 0]

        for ax, w, title in [(row[0], w_cen, 'Centrals'), (row[1], w_sat, 'Satellites')]:
            sel = sorted(m.thin(snap, w, dilute)) if len(w) > dilute else w

            sc = ax.scatter(np.log10(Mvir[sel]), Concentration[sel],
                            c=np.log10(StellarMass[sel]), s=3, alpha=0.4,
                            cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_xlabel(r'$\log_{10} M_{\rm vir}\ [M_{\odot}]$', fontsize=14)
            ax.set_title(f'{m.label}: {title} — z = {m.redshift(snap):.2f}', fontsize=14)
            ax.set_xlim(9.5, 15.0)
            ax.set_ylim(0, 40)

        row[0].set_ylabel(r'Concentration $c$', fontsize=14)
        cbar = fig.colorbar(sc, ax=row[1], pad=0.02)
        cbar.set_label(r'$\log_{10} M_{\star}\ [M_{\odot}]$', fontsize=14)
        cbar.solids.set_alpha(1.0)

    plt.tight_layout()
    save_figure('concentration_vs_mvir', fig, dpi=150)


def local_sfr_vs_halo_accretion_rate(models):
    print('Plotting SFR vs halo accretion rate')

    fig, axes = model_panels(models, sharex=True, sharey=True)

    for ax, m in zip(axes, models):
        snap = m.local_snap
        SFR = m.get(snap, 'SfrDisk') + m.get(snap, 'SfrBulge')
        HaloAccretionRate = m.get(snap, 'mdot_cool') + m.get(snap, 'mdot_stream')

        w = np.where((SFR > 0.0) & (HaloAccretionRate > 0.0))[0]
        if len(w) > dilute:
            w = m.thin(snap, w, dilute)

        log_acc = np.log10(HaloAccretionRate[w])
        log_sfr = np.log10(SFR[w])
        log_mstar = np.log10(np.maximum(m.get(snap, 'StellarMass')[w], 1.0))
        sc = ax.scatter(log_acc, log_sfr, c=log_mstar, s=2, alpha=0.5,
                        cmap='viridis', rasterized=True)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10} M_{\star}\ [M_{\odot}]$')
        cbar.solids.set_alpha(1.0)

        x_ref = np.array([-4.0, 4.0])
        ax.plot(x_ref, x_ref, 'k--', lw=1.2, alpha=0.5, label='SFR = accretion rate')

        ax.set_xlabel(r'$\log_{10}\ \dot{M}_{\rm acc}\ (M_{\odot}\ \mathrm{yr}^{-1})$')
        ax.set_title(f'{m.label}: Centrals — z = {m.redshift(snap):.2f}')
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(-6.0, 4.0)

        leg = ax.legend(loc='upper left')
        leg.draw_frame(False)
        for t in leg.get_texts():
            t.set_fontsize('medium')

    axes[0].set_ylabel(r'$\log_{10}\ \mathrm{SFR}\ (M_{\odot}\ \mathrm{yr}^{-1})$')
    plt.tight_layout()
    save_figure('sfr_vs_halo_accretion_rate', fig, dpi=150)


def local_tcool_vs_mvir(models):
    print('Plotting cooling rate vs Mvir')

    fig, axes = model_panels(models, sharex=True, sharey=True)

    for ax, m in zip(axes, models):
        snap = m.local_snap
        tcool = m.get(snap, 'CoolingRate')
        Mvir = m.get(snap, 'Mvir')
        Rvir = m.get(snap, 'Rvir')
        Regime = m.get(snap, 'Regime')
        Tvir = 35.9 * m.get(snap, 'Vvir')**2  # in Kelvin

        w = np.where((Mvir > 0.0) & (tcool > 0.0) & (Rvir > 0.0))[0]
        log_Mvir = np.log10(Mvir[w])
        tcool = tcool[w]
        Regime = Regime[w]
        Tvir = Tvir[w]

        print(f'  {m.label}: cooled gas rate vs Mvir sample stats:')
        if tcool.size > 0:
            print(f'    cooled gas rate: {tcool.min():.2f} to {tcool.max():.2f}')
        else:
            print('    no positive cooled gas rates found; skipping statistics for this plot')

        cgm = Regime == 0
        hot = Regime == 1
        unclassified = ~(cgm | hot)
        if np.any(cgm):
            ax.scatter(log_Mvir[cgm], tcool[cgm], s=2, alpha=0.5,
                       color='tab:blue', label='CGM regime', rasterized=True)
        if np.any(hot):
            ax.scatter(log_Mvir[hot], tcool[hot], s=2, alpha=0.5,
                       color='tab:red', label='Hot regime', rasterized=True)
        if np.any(unclassified):
            temperature_points = ax.scatter(log_Mvir[unclassified], tcool[unclassified],
                                            s=2, alpha=0.5, c=Tvir[unclassified],
                                            cmap='viridis', norm=LogNorm(),
                                            label='Unclassified', rasterized=True)
            colorbar = fig.colorbar(temperature_points, ax=ax, pad=0.02)
            colorbar.set_label(r'$T_{\rm vir}$ (K)')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10} M_{\rm vir}\ (M_{\odot})$')
        ax.legend(frameon=False, markerscale=5)

    axes[0].set_ylabel(r'$\mathrm{Cooling\ rate}\ (M_{\odot}\ \mathrm{Gyr}^{-1})$')
    save_figure('tcool_vs_mvir', fig, dpi=150)


def local_cgm_fraction_vs_mvir(models):
    print('Plotting CGM fraction vs Mvir')

    plt.figure()
    ax = plt.subplot(111)

    for m in models:
        snap = m.local_snap
        Mvir = m.get(snap, 'Mvir')
        CGMgas = m.get(snap, 'CGMgas')

        with np.errstate(divide='ignore', invalid='ignore'):
            CGM_fraction = np.log10(CGMgas / Mvir) * m.BaryonFrac  # Normalize by cosmic baryon fraction
            finite = np.isfinite(CGM_fraction)

            print(f'  {m.label}: CGM fraction vs Mvir sample stats:')
            if np.any(finite):
                print(f'    CGM fraction: {CGM_fraction[finite].min():.4f} to {CGM_fraction[finite].max():.4f}')

            ax.scatter(np.log10(Mvir), CGM_fraction, s=2, alpha=0.5, color=m.color, label=m.label,
                       rasterized=True)

    ax.set_xlabel(r'$\log_{10} M_{\rm vir}\ (M_{\odot})$')
    ax.set_ylabel(r'$\mathrm{CGM\ Fraction}$')
    ax.set_xlim(10.0, 12.0)
    ax.set_ylim(-2.5, 0.0)
    if len(models) > 1:
        ax.legend(frameon=False, markerscale=5)

    save_figure('cgm_fraction_vs_mvir', dpi=150)


def local_ejection_efficiency_vs_vvir(models):
    print('Plotting ejection efficiency vs Vvir')

    plt.figure()
    ax = plt.subplot(111)

    # SAGE26 FIRE constants
    epsilon_disk = 2.9
    epsilon_halo = 0.3
    V_sn = 501.0  # km/s, characteristic supernova velocity

    plotted = False
    for m in models:
        snap = m.local_snap
        sfr = m.get(snap, 'SfrDisk') + m.get(snap, 'SfrBulge')
        MassLoading = m.get(snap, 'MassLoading')
        Vvir = m.get(snap, 'Vvir')
        w = np.where((sfr > 0.0) & (MassLoading > 0.0) & (Vvir > 0.0) & (m.get(snap, 'Mvir') > 0.0))[0]
        if len(w) == 0:
            print(f'  {m.label}: no galaxies with SFR > 0, MassLoading > 0, and Vvir > 0.')
            continue
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        MassLoading = MassLoading[w]
        Vvir = Vvir[w]
        sfr = sfr[w]
        reheat = MassLoading * sfr

        # Extract the pure scaling factor 'f' from eta_reheat (MassLoading)
        f_factor = MassLoading / epsilon_disk
        # Eq. 4: Feedback Energy (using f_factor, not MassLoading)
        E_fb = 0.5 * epsilon_halo * f_factor * sfr * (V_sn**2)
        # Eq. 5: Lifting Energy (using the full MassLoading / eta_reheat)
        E_lift = 0.5 * reheat * (Vvir**2)
        # Eq. 6: Ejected mass, which cannot be negative
        eject = np.maximum(0.0, (E_fb - E_lift) / (0.5 * Vvir**2))

        print(f'  {m.label}: ejection sample stats:')
        print(f'    Ejection energy: {eject.min():.4f} to {eject.max():.4f}')

        ejection_efficiency = (eject / sfr)
        print(f'    Ejection efficiency: {ejection_efficiency.min():.4f} to {ejection_efficiency.max():.4f}')

        with np.errstate(divide='ignore', invalid='ignore'):
            log_eff = np.log10(ejection_efficiency)
            ax.scatter(np.log10(Vvir), log_eff, s=2, alpha=0.5, color=m.color, rasterized=True)

            # Median line and 1-sigma shading
            bin_edges = np.arange(1.0, 3.0, 0.1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            digitized = np.digitize(np.log10(Vvir), bin_edges)
            medians = np.array([np.median(log_eff[digitized == i]) for i in range(1, len(bin_edges))])
            stds = np.array([np.std(log_eff[digitized == i]) for i in range(1, len(bin_edges))])

        ax.fill_between(bin_centers, medians - stds, medians + stds, color=m.color, alpha=0.2, rasterized=True)
        ax.plot(bin_centers, medians, color=m.color, lw=2, label=f'{m.label}: median ± 1σ')
        plotted = True

    if not plotted:
        print('  Skipping ejection_efficiency_vs_vvir: no galaxies with SFR > 0, MassLoading > 0, and Vvir > 0.\n')
        plt.close()
        return

    ax.set_xlabel(r'$\log_{10} V_{\rm vir}\ (\mathrm{km/s})$')
    ax.set_ylabel(r'$\mathrm{Ejection\ Efficiency}$')
    ax.legend(frameon=False)

    save_figure('ejection_efficiency_vs_vvir', dpi=150)


def local_disk_radius_vs_stellar_mass(models):
    print('Plotting disk radii vs Rvir')

    plt.figure()
    ax = plt.subplot(111)

    rvir_min, rvir_max = np.inf, -np.inf
    for m in models:
        snap = m.local_snap
        # DiskRadius is the exponential scale length of the MMW98 disk (src/model_misc.c).
        # Sizes are measured as effective (half-light/half-mass) radii, so put the model on
        # that scale rather than pulling the observations down onto a scale length:
        # r_e = 1.68 r_d for a pure exponential -- the same constant the model itself uses
        # for this conversion (DISK_HALF_MASS_FRAC in src/model_mergers.c).
        DiskRadius = m.get(snap, 'DiskRadius') * m.UnitLength_kpc * 1.68  # kpc/h, effective radius
        StellarMass = m.get(snap, 'StellarMass')
        Rvir = m.get(snap, 'Rvir') * m.UnitLength_kpc  # kpc/h

        with np.errstate(divide='ignore'):
            logm = np.log10(StellarMass)
        w = np.where((Rvir > 0.0) & (DiskRadius > 0.0) & (logm > 9.3) & (logm < 11.3))[0]
        if len(w) == 0:
            print(f'  {m.label}: no galaxies with StellarMass > 0 and DiskRadius > 0.')
            continue
        if(len(w) > dilute): w = m.thin(snap, w, dilute)

        ax.scatter(Rvir[w], DiskRadius[w], s=2, alpha=0.5, color=m.color, rasterized=True)

        # Median line and 1-sigma shading
        bin_edges = np.logspace(np.log10(Rvir[w].min()), np.log10(Rvir[w].max()), 50)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        digitized = np.digitize(Rvir[w], bin_edges)
        medians = np.array([np.median(DiskRadius[w][digitized == i]) for i in range(1, len(bin_edges))])
        stds = np.array([np.std(DiskRadius[w][digitized == i]) for i in range(1, len(bin_edges))])

        ax.fill_between(bin_centers, medians - stds, medians + stds, color=m.color, alpha=0.2, rasterized=True)
        ax.plot(bin_centers, medians, color=m.color, lw=2,
                label=f'{m.label}: median ± 1σ (matching Somerville+18 selection)')
        rvir_min = min(rvir_min, Rvir[w].min())
        rvir_max = max(rvir_max, Rvir[w].max())

    if not np.isfinite(rvir_min):
        print('  Skipping disk_radius_vs_stellar_mass: no galaxies with StellarMass > 0 and DiskRadius > 0.\n')
        plt.close()
        return

    # Somerville et al. (2018), GAMA at z=0.1: the stellar-to-halo size ratio
    # SRHR = r_e / Rvir, tabulated in eight stellar mass bins from log(M*)=9.25 to 11.34.
    # Taken as published -- the model has already been converted to r_e above.
    # Both radii are kpc/h here, so the ratio carries no h and needs no correction.
    try:
        srhr = np.loadtxt(os.path.join(DataDir, 'SizesAndAM', 'RSRHSomerville.dat'))
        re_over_rvir = srhr[:, 1]
        rvir_ref = np.array([rvir_min, rvir_max])
        ax.fill_between(rvir_ref, re_over_rvir.min() * rvir_ref, re_over_rvir.max() * rvir_ref,
                        color='firebrick', alpha=0.25, zorder=5,
                        label=r'Somerville+18 SRHR (9.3 < $\log M_\star$ < 11.3)')
        ax.plot(rvir_ref, np.median(re_over_rvir) * rvir_ref, '--', color='firebrick',
                lw=2, zorder=6)
    except Exception as e:
        print(f"Warning: Could not load RSRHSomerville.dat: {e}")

    ax.set_xlabel(r'$R_{\rm vir}\ (\mathrm{kpc}/h)$')
    ax.set_ylabel(r'$R_{\rm e,disk}\ (\mathrm{kpc}/h)$')
    ax.set_ylim(10**-2, 10**2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', frameon=False, fontsize=11)

    save_figure('disk_radius_vs_stellar_mass', dpi=150)


def local_half_mass_radius_vs_stellar_mass(models):
    print('Plotting half-mass radius vs stellar mass coloured by sSFR')

    fig, axes = model_panels(models, sharex=True, sharey=True)

    for ax, m in zip(axes, models):
        snap = m.local_snap
        # SAGE26 has no light, so this is the stellar half-mass radius. The galaxy size is the
        # mass-weighted mean of the disk and bulge half-mass radii, as Shark defines a galaxy's
        # composite size (Lagos et al. 2018; shark Galaxy::composite_size). DiskRadius is the
        # exponential scale length, r_e = 1.68 r_d, as in the disk-size plot above.
        DiskRadius = m.get(snap, 'DiskRadius') * m.UnitLength_kpc * 1.68  # kpc/h, half-mass
        BulgeRadius = m.get(snap, 'BulgeRadius') * m.UnitLength_kpc        # kpc/h, half-mass
        StellarMass = m.get(snap, 'StellarMass')
        BulgeMass = m.get(snap, 'BulgeMass')
        SFR = m.get(snap, 'SfrDisk') + m.get(snap, 'SfrBulge')  # Msun/yr

        w = np.where((StellarMass > 1.0e8) & ((DiskRadius > 0.0) | (BulgeRadius > 0.0)))[0]
        if len(w) == 0:
            print(f'  {m.label}: no galaxies with M* > 1e8 and a size.')
            continue
        if len(w) > dilute: w = m.thin(snap, w, dilute)
        w = np.array(w)
        DiskMass = StellarMass[w] - BulgeMass[w]
        half_mass_radius = (DiskMass * DiskRadius[w] + BulgeMass[w] * BulgeRadius[w]) / StellarMass[w]
        # Galaxies with no star formation sit at the floor of the colour scale
        log_ssfr = np.log10(np.maximum(SFR[w] / StellarMass[w], 1.0e-13))

        sc = ax.scatter(StellarMass[w], half_mass_radius, c=log_ssfr, cmap='jet_r',
                        vmin=-13, vmax=-9, s=2, alpha=0.5, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}\,\mathrm{sSFR}\ (\mathrm{yr}^{-1})$')
        cbar.solids.set_alpha(1.0)

        ax.set_xlabel(r'$M_{\rm stars}\ (M_{\odot})$')
        ax.set_xscale('log')
        ax.set_yscale('log')

    axes[0].set_ylabel(r'$R_{\rm 50}\ (\mathrm{kpc}/h)$')
    save_figure('half_mass_radius_vs_stellar_mass', fig, dpi=150)


LOCAL_PLOTS_B = [
    local_baryon_fraction,
    local_mass_reservoir_scatter,
    local_spatial_distribution,
    local_spatial_distribution_3d_box,
    local_star_formation_rate,
    local_mass_loading_vs_vvir,
    local_stellar_vs_halo_mass_by_regime,
    local_ssfr_vs_stellar_mass,
    local_concentration_vs_mvir,
    local_sfr_vs_halo_accretion_rate,
    local_tcool_vs_mvir,
    local_cgm_fraction_vs_mvir,
    local_ejection_efficiency_vs_vvir,
    local_disk_radius_vs_stellar_mass,
    local_half_mass_radius_vs_stellar_mass,
]

# HISTORY PLOTS (redshift evolution)
# ==================================================================

# ========================== HISTORY PLOTS (A) ==========================

# Redshift bins shared by the binned-evolution plots
HISTORY_Z_BINS = [
    (0.2, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 2.0),
    (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.5), (4.5, 5.5),
    (5.5, 6.5), (6.5, 7.5), (7.5, 8.5), (8.5, 10.0), (10.0, 12.0)
]


def history_snaps_in_z_range(m, z_min, z_max):
    """This model's history snapshots with z_min <= z <= z_max."""
    return [s for s in find_snapshots_in_z_range(z_min, z_max, m.redshifts) if s in m.history_snaps]


def add_model_linestyle_legend(ax, models, loc='upper right', fontsize=10):
    """Second legend naming the models by linestyle, kept alongside the existing one."""
    if len(models) < 2:
        return
    first = ax.get_legend()
    leg = ax.legend(handles=model_legend_handles(models, linestyles=True), loc=loc,
                    fontsize=fontsize, frameon=False)
    if first is not None:
        ax.add_artist(first)
    return leg


def history_rcool_to_rvir_above_mshock(models):
    # ====================================================================
    # MEASUREMENT: fraction of Mvir > Mshock haloes that have rcool > Rvir
    # --------------------------------------------------------------------
    # Mshock is the De Lucia & Blaizot (2006) eq. 38 virial-shock mass scale
    # hardcoded in src/model_cooling_heating.c (MSHOCK_DB06_MSUN = 6e11 Msun,
    # physical). Above it a stable virial shock can develop, so naively those
    # haloes should cool in the "hot mode". The diagnostic rcool/Rvir is stored
    # per galaxy as RcoolToRvir: rcool > Rvir flags rapid/whole-halo (cold-mode)
    # cooling even though the halo is massive enough to shock-heat.
    #
    # Caveat on the cap: in the regime-aware cooling path (CGMrecipeOn=1, the
    # default) solve_for_rcool() clamps rcool <= Rvir, so those galaxies pin at
    # exactly RcoolToRvir = 1.0. Hot-regime haloes cooled via cooling_recipe_hot
    # keep an uncapped value that can exceed 1. We therefore report both the
    # strict rcool > Rvir fraction and the rcool >= Rvir (>= 1.0) fraction.
    # RcoolToRvir < 0 is the sentinel for "undefined" and is excluded.
    # ====================================================================
    print('Measuring rcool > Rvir fraction for Mvir > Mshock haloes\n')

    measure_targets = [0.0, 2.0, 6.0]

    results = {}
    for m in models:
        Mshock_Msun = m.MShockMsun  # physical Msun, the run's MShockMsun parameter
        print(f'  [{m.label}]  Mshock = {Mshock_Msun:.2e} Msun (De Lucia & Blaizot 2006 eq. 38)')
        print(f'  {"z":>6} {"snap":>5} {"N(Mvir>Mshock)":>15} '
              f'{"f(rcool>Rvir)":>14} {"f(rcool>=Rvir)":>15}')
        frac_strict = []
        actual_z = []
        selections = []
        for target_z in measure_targets:
            snap = int(np.argmin(np.abs(np.array(m.redshifts) - target_z)))
            z_here = m.redshifts[snap]
            actual_z.append(z_here)

            if snap not in m.history_snaps or np.size(m.get(snap, 'Mvir')) == 0:
                print(f'  {target_z:>6.1f} {snap:>5} {"--":>15} {"--":>14} {"--":>15}')
                frac_strict.append(np.nan)
                selections.append(None)
                continue

            Mvir = m.get(snap, 'Mvir')                     # physical Msun
            rcool_ratio = m.get(snap, 'RcoolToRvir')       # rcool / Rvir (sentinel < 0 = undefined)

            sel = (Mvir > Mshock_Msun) & (rcool_ratio >= 0.0)
            N = int(np.sum(sel))
            if N == 0:
                print(f'  {z_here:>6.2f} {snap:>5} {0:>15} {"--":>14} {"--":>15}')
                frac_strict.append(np.nan)
                selections.append(None)
                continue

            f_strict = np.mean(rcool_ratio[sel] > 1.0)
            f_geq = np.mean(rcool_ratio[sel] >= 1.0)
            frac_strict.append(f_strict)
            selections.append(rcool_ratio[sel])
            print(f'  {z_here:>6.2f} {snap:>5} {N:>15} {f_strict:>14.3f} {f_geq:>15.3f}')
        results[m.label] = (frac_strict, actual_z, selections)
        print()

    # Quick visual: rcool/Rvir distribution for the Mvir > Mshock subset
    plt.figure()
    ax = plt.subplot(111)
    rr_bins = np.arange(0.0, 2.2, 0.05)
    colors_m = plt.cm.viridis(np.linspace(0.15, 0.85, len(measure_targets)))
    for m in models:
        frac_strict, actual_z, selections = results[m.label]
        for k in range(len(measure_targets)):
            if selections[k] is None:
                continue
            label = f'z = {actual_z[k]:.1f}  (N={len(selections[k])}, f$_{{>1}}$={frac_strict[k]:.2f})'
            if len(models) > 1:
                label = f'{m.label}: ' + label
            ax.hist(np.clip(selections[k], rr_bins[0], rr_bins[-1]), bins=rr_bins,
                    histtype='step', lw=2, density=True, color=colors_m[k], ls=m.ls,
                    label=label)
    ax.axvline(1.0, color='k', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel(r'$r_{\rm cool}\,/\,R_{\rm vir}$', fontsize=14)
    ax.set_ylabel('PDF', fontsize=14)
    ax.set_title(r'$M_{\rm vir} > M_{\rm shock}\ (6\times10^{11}\,M_\odot)$ haloes',
                 fontsize=13)
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    plt.tight_layout()
    save_figure('RcoolToRvir_above_Mshock')


def history_stellar_mass_function_z(models):
    print('Plotting the stellar mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    # Marchesini et al. 2009ApJ...701.1765M SMF, h=0.7

    M = np.arange(7.0, 11.8, 0.01)
    Mstar = np.log10(10.0**10.96)
    alpha = -1.18
    phistar = 30.87*1e-4
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    if(whichimf == 0):
        plt.plot(np.log10(10.0**M *1.6), yval, ':', lw=10, alpha=0.5, label='Marchesini et al. 2009 z=[0.1]')
    elif(whichimf == 1):
        plt.plot(np.log10(10.0**M *1.6 /1.8), yval, ':', lw=10, alpha=0.5, label='Marchesini et al. 2009 z=[0.1]')

    M = np.arange(9.3, 11.8, 0.01)
    Mstar = np.log10(10.0**10.91)
    alpha = -0.99
    phistar = 10.17*1e-4
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    if(whichimf == 0):
        plt.plot(np.log10(10.0**M *1.6), yval, 'b:', lw=10, alpha=0.5, label='... z=[1.3,2.0]')
    elif(whichimf == 1):
        plt.plot(np.log10(10.0**M *1.6/1.8), yval, 'b:', lw=10, alpha=0.5, label='... z=[1.3,2.0]')

    M = np.arange(9.7, 11.8, 0.01)
    Mstar = np.log10(10.0**10.96)
    alpha = -1.01
    phistar = 3.95*1e-4
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    if(whichimf == 0):
        plt.plot(np.log10(10.0**M *1.6), yval, 'g:', lw=10, alpha=0.5, label='... z=[2.0,3.0]')
    elif(whichimf == 1):
        plt.plot(np.log10(10.0**M *1.6/1.8), yval, 'g:', lw=10, alpha=0.5, label='... z=[2.0,3.0]')

    M = np.arange(10.0, 11.8, 0.01)
    Mstar = np.log10(10.0**11.38)
    alpha = -1.39
    phistar = 0.53*1e-4
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    if(whichimf == 0):
        plt.plot(np.log10(10.0**M *1.6), yval, 'r:', lw=10, alpha=0.5, label='... z=[3.0,4.0]')
    elif(whichimf == 1):
        plt.plot(np.log10(10.0**M *1.6/1.8), yval, 'r:', lw=10, alpha=0.5, label='... z=[3.0,4.0]')

    # Model: z=0, 1.3, 2, 3 in black, blue, green, red; models told apart by linestyle
    snap_colors = ['k', 'b', 'g', 'r']
    binwidth = 0.1
    for mi_, m in enumerate(models):
        # Snapshot closest to each target z (the Millennium snaps 37, 32, 27), so every
        # simulation is drawn at matching redshifts
        SMFsnaps = [m.LastSnap] + [int(np.argmin(np.abs(np.array(m.redshifts) - target_z)))
                                   for target_z in (1.39, 2.07, 3.06)]

        for k, snap in enumerate(SMFsnaps):
            if snap not in m.history_snaps:
                continue
            sm = m.get(snap, 'StellarMass')
            w = np.where(sm > 0.0)[0]
            if len(w) == 0:
                continue
            mass = np.log10(sm[w])

            mi = np.floor(min(mass)) - 2
            ma = np.floor(max(mass)) + 2
            NB = int((ma - mi) / binwidth)
            (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
            xaxeshisto = binedges[:-1] + 0.5 * binwidth

            label = 'Model galaxies' if (k == 0 and mi_ == 0) else None
            plt.plot(xaxeshisto, counts / m.volume / binwidth, color=snap_colors[k], ls=m.ls, label=label)

    plt.yscale('log')
    plt.axis([7.0, 12.2, 1.0e-6, 1.0e-1])

    # Set the x-axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}$)')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
    add_model_linestyle_legend(ax, models, loc='upper right', fontsize='medium')

    plt.tight_layout()
    save_figure('StellarMassFunction_z')


def history_sfr_density(models):
    print('Plotting SFR density evolution for all galaxies')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    ObsSFRdensity = np.array([
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
        ], dtype=np.float32)

    ObsRedshift = ObsSFRdensity[:, 0]
    xErrLo = np.abs(ObsSFRdensity[:, 0]-ObsSFRdensity[:, 2])
    xErrHi = np.abs(ObsSFRdensity[:, 3]-ObsSFRdensity[:, 0])

    ObsSFR = np.log10(ObsSFRdensity[:, 1])
    yErrLo = np.abs(np.log10(ObsSFRdensity[:, 1])-np.log10(ObsSFRdensity[:, 4]))
    yErrHi = np.abs(np.log10(ObsSFRdensity[:, 5])-np.log10(ObsSFRdensity[:, 1]))

    # plot observational data (compilation used in Croton et al. 2006)
    plt.errorbar(ObsRedshift, ObsSFR, yerr=[yErrLo, yErrHi], xerr=[xErrLo, xErrHi], color='g', lw=1.0, alpha=0.5, marker='o', ls='none', label='Observations')

    for m in models:
        snaps = m.history_snaps
        SFR_density = np.array([np.sum(m.get(s, 'SfrDisk') + m.get(s, 'SfrBulge')) / m.volume
                                for s in snaps])
        z = np.array(m.redshifts[snaps])
        nonzero = np.where(SFR_density > 0.0)[0]
        plt.plot(z[nonzero], np.log10(SFR_density[nonzero]), color=m.color, lw=3.0, label=m.label)

    plt.ylabel(r'$\log_{10} \mathrm{SFR\ density}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$')  # Set the y...
    plt.xlabel(r'$\mathrm{redshift}$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.axis([0.0, 8.0, -3.0, -0.4])
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    save_figure('History-SFR-density')


def history_stellar_mass_density(models):
    print('Plotting stellar mass density evolution')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    # SMD observations taken from Marchesini+ 2009, h=0.7
    # Values are (minz, maxz, rho,-err,+err)

    dickenson2003 = np.array(((0.6,1.4,8.26,0.08,0.08),
                     (1.4,2.0,7.86,0.22,0.33),
                     (2.0,2.5,7.58,0.29,0.54),
                     (2.5,3.0,7.52,0.51,0.48)),float)
    drory2005 = np.array(((0.25,0.75,8.3,0.15,0.15),
                (0.75,1.25,8.16,0.15,0.15),
                (1.25,1.75,8.0,0.16,0.16),
                (1.75,2.25,7.85,0.2,0.2),
                (2.25,3.0,7.75,0.2,0.2),
                (3.0,4.0,7.58,0.2,0.2)),float)
    PerezGonzalez2008 = np.array(((0.2,0.4,8.41,0.06,0.06),
             (0.4,0.6,8.37,0.04,0.04),
             (0.6,0.8,8.32,0.05,0.05),
             (0.8,1.0,8.24,0.05,0.05),
             (1.0,1.3,8.15,0.05,0.05),
             (1.3,1.6,7.95,0.07,0.07),
             (1.6,2.0,7.82,0.07,0.07),
             (2.0,2.5,7.67,0.08,0.08),
             (2.5,3.0,7.56,0.18,0.18),
             (3.0,3.5,7.43,0.14,0.14),
             (3.5,4.0,7.29,0.13,0.13)),float)
    glazebrook2004 = np.array(((0.8,1.1,7.98,0.14,0.1),
                     (1.1,1.3,7.62,0.14,0.11),
                     (1.3,1.6,7.9,0.14,0.14),
                     (1.6,2.0,7.49,0.14,0.12)),float)
    fontana2006 = np.array(((0.4,0.6,8.26,0.03,0.03),
                  (0.6,0.8,8.17,0.02,0.02),
                  (0.8,1.0,8.09,0.03,0.03),
                  (1.0,1.3,7.98,0.02,0.02),
                  (1.3,1.6,7.87,0.05,0.05),
                  (1.6,2.0,7.74,0.04,0.04),
                  (2.0,3.0,7.48,0.04,0.04),
                  (3.0,4.0,7.07,0.15,0.11)),float)
    rudnick2006 = np.array(((0.0,1.0,8.17,0.27,0.05),
                  (1.0,1.6,7.99,0.32,0.05),
                  (1.6,2.4,7.88,0.34,0.09),
                  (2.4,3.2,7.71,0.43,0.08)),float)
    elsner2008 = np.array(((0.25,0.75,8.37,0.03,0.03),
                 (0.75,1.25,8.17,0.02,0.02),
                 (1.25,1.75,8.02,0.03,0.03),
                 (1.75,2.25,7.9,0.04,0.04),
                 (2.25,3.0,7.73,0.04,0.04),
                 (3.0,4.0,7.39,0.05,0.05)),float)

    obs = (dickenson2003,drory2005,PerezGonzalez2008,glazebrook2004,fontana2006,rudnick2006,elsner2008)

    for o in obs:
        xval = ((o[:,1]-o[:,0])/2.)+o[:,0]
        if(whichimf == 0):
            ax.errorbar(xval, np.log10(10**o[:,2] *1.6), xerr=(xval-o[:,0], o[:,1]-xval), yerr=(o[:,3], o[:,4]), alpha=0.5, lw=1.0, marker='o', ls='none')
        elif(whichimf == 1):
            ax.errorbar(xval, np.log10(10**o[:,2] *1.6/1.8), xerr=(xval-o[:,0], o[:,1]-xval), yerr=(o[:,3], o[:,4]), alpha=0.5, lw=1.0, marker='o', ls='none')

    for m in models:
        snaps = m.history_snaps
        smd = np.zeros(len(snaps))
        for k, snap in enumerate(snaps):
            sm = m.get(snap, 'StellarMass')
            w = np.where((sm > 1.0e8) & (sm < 1.0e13))[0]
            if(len(w) > 0):
                smd[k] = np.sum(sm[w]) / m.volume

        z = np.array(m.redshifts[snaps])
        nonzero = np.where(smd > 0.0)[0]
        plt.plot(z[nonzero], np.log10(smd[nonzero]), color=m.color, ls='-', lw=3.0, label=m.label)

    plt.ylabel(r'$\log_{10}\ \phi\ (M_{\odot}\ \mathrm{Mpc}^{-3})$')  # Set the y...
    plt.xlabel(r'$\mathrm{redshift}$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.axis([0.0, 4.2, 6.5, 9.0])
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    save_figure('History-stellar-mass-density')


def history_sfr_evolution(models):
    print('Plotting SFR evolution with redshift bins')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS

    # Define colormap - from dark to light/orange
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    for mi_, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            # Use same mass bins as above
            if i >= len(z_bins) - 4:
                mass_bins = np.arange(8.0, 12.5, 0.1)
            else:
                mass_bins = np.arange(7.2, 12.5, 0.1)
            mass_centers = mass_bins[:-1] + 0.05

            sfr_snapshots = []
            for snap_idx in snap_indices:
                # Only consider galaxies with positive stellar mass and SFR
                sm = m.get(snap_idx, 'StellarMass')
                sfr_total = m.get(snap_idx, 'SfrDisk') + m.get(snap_idx, 'SfrBulge')
                w = np.where((sm > 0.0) & (sfr_total > 0.0))[0]
                if len(w) > 0:
                    masses = np.log10(sm[w])
                    sfrs = np.log10(sfr_total[w])
                    # Bin by stellar mass
                    sfr_bin_means = np.full(len(mass_centers), np.nan)
                    sfr_bin_stds = np.full(len(mass_centers), np.nan)
                    for j in range(len(mass_centers)):
                        mask = (masses >= mass_bins[j]) & (masses < mass_bins[j+1])
                        if np.sum(mask) > 0:
                            sfr_bin_means[j] = np.mean(sfrs[mask])
                            sfr_bin_stds[j] = np.std(sfrs[mask])
                    sfr_snapshots.append((sfr_bin_means, sfr_bin_stds))
            if len(sfr_snapshots) == 0:
                continue
            # Stack and compute mean and error
            sfr_bin_means_stack = np.array([x[0] for x in sfr_snapshots])
            sfr_mean = np.nanmean(sfr_bin_means_stack, axis=0)
            sfr_err = np.nanstd(sfr_bin_means_stack, axis=0) / np.sqrt(len(sfr_bin_means_stack))
            valid = ~np.isnan(sfr_mean) & ~np.isnan(sfr_err)
            if not np.any(valid):
                continue
            label = f'{z_min:.1f} < z < {z_max:.1f}' if mi_ == 0 else None
            ax.plot(mass_centers[valid], sfr_mean[valid], color=colors[i], linewidth=2, ls=m.ls, label=label)
            # Error bands on the first model only, so overlaid models stay readable
            if mi_ == 0:
                ax.fill_between(mass_centers[valid], sfr_mean[valid] - sfr_err[valid], sfr_mean[valid] + sfr_err[valid], color=colors[i], alpha=0.3)

    ax.set_xlim(8.0, 12.2)
    ax.set_ylim(-4, 3)
    ax.set_xlabel(r'$\log_{10} M_* [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\log_{{10}}\mathrm{{SFR}}\ [M_\odot/yr]$', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    leg = ax.legend(loc='lower right', fontsize=10, frameon=False, ncol=3)
    for text in leg.get_texts():
        text.set_fontsize(10)
    add_model_linestyle_legend(ax, models, loc='upper left')

    plt.tight_layout()
    save_figure('SFR_evolution', dpi=300, bbox_inches='tight')


def history_stellar_mass_halo_mass_relation(models):
    print('Plotting stellar mass-halo mass relation evolution')

    plt.figure()
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS

    # Colours mark the redshift bin. The original also cycled linestyles across the bins;
    # here the linestyle names the model instead.
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    # Halo mass bins
    halo_mass_bins = np.arange(10.0, 15.0, 0.1)
    halo_mass_centers = halo_mass_bins[:-1] + 0.05

    for mi_, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            # Calculate SMHM relation for each snapshot separately
            stellar_mass_binned = []

            for snap_idx in snap_indices:
                sm = m.get(snap_idx, 'StellarMass')
                mvir = m.get(snap_idx, 'Mvir')
                # Get valid galaxies (positive stellar and halo masses)
                w = np.where((sm > 0.0) & (mvir > 0.0))[0]

                if len(w) > 0:
                    stellar_masses = sm[w]  # Keep in linear units
                    halo_masses = np.log10(mvir[w])

                    # Bin by halo mass and calculate mean stellar mass in each bin
                    stellar_mass_in_bins = np.full(len(halo_mass_centers), np.nan)
                    for j in range(len(halo_mass_centers)):
                        bin_mask = (halo_masses >= halo_mass_bins[j]) & (halo_masses < halo_mass_bins[j+1])
                        if np.sum(bin_mask) > 0:
                            stellar_mass_in_bins[j] = np.mean(stellar_masses[bin_mask])

                    stellar_mass_binned.append(stellar_mass_in_bins)

            if len(stellar_mass_binned) == 0:
                continue

            stellar_mass_binned = np.array(stellar_mass_binned)

            # Calculate mean and 1-sigma error across snapshots
            stellar_mass_mean = np.nanmean(stellar_mass_binned, axis=0)
            stellar_mass_err = np.nanstd(stellar_mass_binned, axis=0) / np.sqrt(len(stellar_mass_binned))

            # Only plot where we have valid data
            valid = ~np.isnan(stellar_mass_mean) & ~np.isnan(stellar_mass_err)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if mi_ == 0 else None
            ax.plot(halo_mass_centers[valid], np.log10(stellar_mass_mean[valid]),
                    color=colors[i], linewidth=2, linestyle=m.ls, label=label)

            # Error bands on the first model only, so overlaid models stay readable
            if mi_ == 0:
                ax.fill_between(halo_mass_centers[valid],
                               np.log10(stellar_mass_mean[valid] - stellar_mass_err[valid]),
                               np.log10(stellar_mass_mean[valid] + stellar_mass_err[valid]),
                               color=colors[i], alpha=0.3)

    ax.set_xlim(10.0, 15.0)
    ax.set_ylim(7.0, 12.0)

    ax.set_xlabel(r'$\log_{10} M_{\rm vir}$ [$M_\odot$]', fontsize=14)
    ax.set_ylabel(r'$\log_{10} M_*$ [$M_\odot$]', fontsize=14)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    leg = ax.legend(loc='lower right', fontsize=10, frameon=False, ncol=2)
    for text in leg.get_texts():
        text.set_fontsize(10)
    add_model_linestyle_legend(ax, models, loc='upper left')

    plt.tight_layout()
    save_figure('StellarMassHaloMassRelation', dpi=300, bbox_inches='tight')


def history_stellar_mass_function_evolution(models):
    print('Plotting stellar mass function evolution with redshift bins')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS

    # Define colormap - from dark to light/orange
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    for mi_, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            # Use different mass bins for the last 4 redshift bins (highest z)
            if i >= len(z_bins) - 4:
                mass_bins = np.arange(8.0, 12.5, 0.1)
            else:
                mass_bins = np.arange(7.2, 12.5, 0.1)

            mass_centers = mass_bins[:-1] + 0.05

            # Calculate stellar mass function for each snapshot separately
            phi_snapshots = []
            bin_width = mass_bins[1] - mass_bins[0]

            for snap_idx in snap_indices:
                sm = m.get(snap_idx, 'StellarMass')
                # Halo-mass minimum is MVIR_CUT, already applied by Model.get
                w = np.where(sm > 0.0)[0]
                if len(w) > 0:
                    masses = np.log10(sm[w])
                    counts, bin_edges = np.histogram(masses, bins=mass_bins)
                    phi_snapshots.append(counts / (m.volume * bin_width))

            if len(phi_snapshots) == 0:
                continue

            phi_snapshots = np.array(phi_snapshots)

            # Calculate mean and 1-sigma error (standard error of the mean)
            phi = np.mean(phi_snapshots, axis=0)
            phi_err = np.std(phi_snapshots, axis=0) / np.sqrt(len(phi_snapshots))

            # Only plot where we have data
            valid = (phi > 0) & (phi_err > 0)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if mi_ == 0 else None
            ax.plot(mass_centers[valid], np.log10(phi[valid]),
                    color=colors[i], linewidth=2, ls=m.ls, label=label)

            # Error bands on the first model only, so overlaid models stay readable
            if mi_ == 0:
                ax.fill_between(mass_centers[valid],
                               np.log10(phi[valid] - phi_err[valid]),
                               np.log10(phi[valid] + phi_err[valid]),
                               color=colors[i], alpha=0.3)

    ax.set_xlim(7.2, 12.0)
    ax.set_ylim(-6, -1)

    ax.set_xlabel(r'$\log_{10} M_* [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}\phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    leg = ax.legend(loc='lower left', fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontsize(10)
    add_model_linestyle_legend(ax, models, loc='upper right')

    plt.tight_layout()
    save_figure('StellarMassFunctionEvolution', dpi=300, bbox_inches='tight')


def history_sfr_function_evolution(models):
    print('Plotting star formation rate function evolution with redshift bins')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    sfr_bins = np.arange(-3.0, 3.5, 0.2)
    sfr_centers = sfr_bins[:-1] + 0.1
    bin_width = sfr_bins[1] - sfr_bins[0]

    for mi_, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            phi_snapshots = []
            for snap_idx in snap_indices:
                sfr_total = m.get(snap_idx, 'SfrDisk') + m.get(snap_idx, 'SfrBulge')
                w = np.where(sfr_total > 0.0)[0]
                if len(w) > 0:
                    log_sfr = np.log10(sfr_total[w])
                    counts, _ = np.histogram(log_sfr, bins=sfr_bins)
                    phi_snapshots.append(counts / (m.volume * bin_width))

            if len(phi_snapshots) == 0:
                continue

            phi_snapshots = np.array(phi_snapshots)
            phi = np.mean(phi_snapshots, axis=0)
            phi_err = np.std(phi_snapshots, axis=0) / np.sqrt(len(phi_snapshots))

            valid = (phi > 0) & (phi_err > 0)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if mi_ == 0 else None
            ax.plot(sfr_centers[valid], np.log10(phi[valid]),
                    color=colors[i], linewidth=2, ls=m.ls, label=label)
            # Error bands on the first model only, so overlaid models stay readable
            if mi_ == 0:
                ax.fill_between(sfr_centers[valid],
                                np.log10(phi[valid] - phi_err[valid]),
                                np.log10(phi[valid] + phi_err[valid]),
                                color=colors[i], alpha=0.3)

    ax.set_xlim(-3.0, 3.2)
    ax.set_ylim(-6, -1)
    ax.set_xlabel(r'$\log_{10}\,\mathrm{SFR}\ [M_\odot\,\mathrm{yr}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}\phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    leg = ax.legend(loc='upper right', fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontsize(10)
    add_model_linestyle_legend(ax, models, loc='lower left')

    plt.tight_layout()
    save_figure('SFRFunctionEvolution', dpi=300, bbox_inches='tight')


def history_black_hole_mass_function(models):
    # Black Hole Mass Function at specific redshifts
    print('Plotting black hole mass function at specific redshifts')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Define redshifts to plot (these correspond to the observation files)
    bhmf_redshifts = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]

    # Define colormap - plasma from dark to light
    colors_bhmf = plt.cm.plasma(np.linspace(0.1, 0.9, len(bhmf_redshifts)))

    # Define mass bins for BHMF
    bhmf_mass_bins = np.arange(6.0, 11.5, 0.1)
    bhmf_mass_centers = bhmf_mass_bins[:-1] + 0.05
    bin_width = bhmf_mass_bins[1] - bhmf_mass_bins[0]

    # Observations are dashed here, so the models skip '--'
    model_ls = [ls for ls in MODEL_LINESTYLES if ls != '--'] or ['-']

    # Plot SAGE model predictions for each redshift
    for mi_, m in enumerate(models):
        ls = model_ls[mi_ % len(model_ls)]
        for i, target_z in enumerate(bhmf_redshifts):
            # Map redshift to closest snapshot
            snap_idx = int(np.argmin(np.abs(np.array(m.redshifts) - target_z)))
            if snap_idx not in m.history_snaps:
                continue
            actual_z = m.redshifts[snap_idx]
            bh = m.get(snap_idx, 'BlackHoleMass')
            # Filter for galaxies with black holes
            w = np.where(bh > 0.0)[0]

            if len(w) > 0:
                bh_masses = np.log10(bh[w])
                counts, bin_edges = np.histogram(bh_masses, bins=bhmf_mass_bins)
                phi = counts / (m.volume * bin_width)

                # Only plot where we have data
                valid = phi > 0
                if np.any(valid):
                    name = 'SAGE' if len(models) == 1 else m.label
                    label = f'z = {actual_z:.1f} ({name})'
                    ax.plot(bhmf_mass_centers[valid], phi[valid],
                           color=colors_bhmf[i], linewidth=2, linestyle=ls, label=label)

    # Load and plot observational data
    obs_files = {
        0.1: 'bh/fig4_bhmf_z0.1.txt',
        1.0: 'bh/fig4_bhmf_z1.0.txt',
        2.0: 'bh/fig4_bhmf_z2.0.txt',
        4.0: 'bh/fig4_bhmf_z4.0.txt',
        6.0: 'bh/fig4_bhmf_z6.0.txt',
        8.0: 'bh/fig4_bhmf_z8.0.txt'
    }

    for i, target_z in enumerate(bhmf_redshifts):
        if target_z in obs_files:
            obs_file = os.path.join(DataDir, obs_files[target_z])
            try:
                # Load observation data (skip header lines starting with #)
                obs_data = np.loadtxt(obs_file)
                obs_mass = obs_data[:, 0]     # log10(Mbh [Msun])
                obs_phi = obs_data[:, 1]      # BHMF_best [Mpc^-3 dex^-1]
                obs_phi_16th = obs_data[:, 2] # BHMF_16th [Mpc^-3 dex^-1]
                obs_phi_84th = obs_data[:, 3] # BHMF_84th [Mpc^-3 dex^-1]

                # Plot observations with dashed line
                label = f'z = {target_z:.1f} (Obs)'
                ax.plot(obs_mass, obs_phi, color=colors_bhmf[i],
                       linewidth=2, linestyle='--', label=label, alpha=0.8)

                # Add shaded error region for observations
                ax.fill_between(obs_mass, obs_phi_16th, obs_phi_84th,
                               color=colors_bhmf[i], alpha=0.2)
            except Exception as e:
                print(f'Warning: Could not load {obs_file}: {e}')

    ax.set_yscale('log')
    ax.set_xlim(6.0, 11.0)
    ax.set_ylim(1e-5, 1e-1)

    ax.set_xlabel(r'$\log_{10} M_{\rm BH} [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    leg = ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)
    for text in leg.get_texts():
        text.set_fontsize(9)

    plt.tight_layout()
    save_figure('BlackHoleMassFunction', dpi=300, bbox_inches='tight')


HISTORY_PLOTS_A = [
    history_rcool_to_rvir_above_mshock,
    history_stellar_mass_function_z,
    history_sfr_density,
    history_stellar_mass_density,
    history_sfr_evolution,
    history_stellar_mass_halo_mass_relation,
    history_stellar_mass_function_evolution,
    history_sfr_function_evolution,
    history_black_hole_mass_function,
]


# --------------------------------------------------------
# History plots, part B: quenched fractions, HMF, concentration, metallicity, H2,
# star formation efficiency, central/satellite counts and fractions
# --------------------------------------------------------

def history_add_model_legend(ax, models, loc='lower right', fontsize=10):
    """Second legend naming the models by linestyle, kept alongside the existing one."""
    if len(models) < 2:
        return
    first = ax.get_legend()
    ax.legend(handles=model_legend_handles(models, linestyles=True), loc=loc,
              fontsize=fontsize, frameon=False)
    if first is not None:
        ax.add_artist(first)


def history_quenched_fraction_evolution(models):
    # Quenched fraction as a function of StellarMass across the same redshift bins (centrals/satellites separated)
    print('Plotting quenched fraction as a function of StellarMass evolution with redshift bins (centrals/satellites)')

    z_bins = HISTORY_Z_BINS
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    # Centrals and satellites are already told apart by linestyle, so each model gets a panel
    fig, axes = model_panels(models, sharey=True)

    for ax, m in zip(axes, models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            if i >= len(z_bins) - 4:
                mass_bins = np.arange(8.0, 13.5, 0.1)
            else:
                mass_bins = np.arange(7.2, 13.5, 0.1)
            mass_centers = mass_bins[:-1] + 0.05

            all_snapshots = []
            central_snapshots = []
            satellite_snapshots = []
            for snap_idx in snap_indices:
                sfr_total = m.get(snap_idx, 'SfrDisk') + m.get(snap_idx, 'SfrBulge')
                stellar_mass = m.get(snap_idx, 'StellarMass')
                galtype = m.get(snap_idx, 'Type')

                # Include ALL galaxies with positive stellar mass (like the local script)
                w = np.where(stellar_mass > 0.0)[0]
                if len(w) > 0:
                    masses = np.log10(stellar_mass[w])
                    sSFR_linear = sfr_total[w] / stellar_mass[w]
                    gtype = galtype[w]
                    quenched = sSFR_linear < 10.0**sSFRcut

                    all_frac = np.full(len(mass_centers), np.nan)
                    cen_frac = np.full(len(mass_centers), np.nan)
                    sat_frac = np.full(len(mass_centers), np.nan)
                    for j in range(len(mass_centers)):
                        in_bin = (masses >= mass_bins[j]) & (masses < mass_bins[j+1])
                        if np.sum(in_bin) > 0:
                            all_frac[j] = np.sum(quenched[in_bin]) / np.sum(in_bin)
                        mask = in_bin & (gtype == 0)
                        if np.sum(mask) > 0:
                            cen_frac[j] = np.sum(quenched[mask]) / np.sum(mask)
                        mask = in_bin & (gtype == 1)
                        if np.sum(mask) > 0:
                            sat_frac[j] = np.sum(quenched[mask]) / np.sum(mask)
                    all_snapshots.append(all_frac)
                    central_snapshots.append(cen_frac)
                    satellite_snapshots.append(sat_frac)

            if len(all_snapshots) == 0:
                continue

            central_snapshots_arr = np.array(central_snapshots)
            cen_frac_mean = np.nanmean(central_snapshots_arr, axis=0)
            sat_frac_mean = np.nanmean(np.array(satellite_snapshots), axis=0)

            # 1-sigma for centrals
            cen_frac_std = np.nanstd(central_snapshots_arr, axis=0)
            valid_cen = ~np.isnan(cen_frac_mean)
            valid_sat = ~np.isnan(sat_frac_mean)

            label = f'{z_min:.1f} < z < {z_max:.1f}'

            # Plot centrals with 1-sigma shading
            ax.plot(mass_centers[valid_cen], cen_frac_mean[valid_cen], color=colors[i], linestyle='-', linewidth=2, label=label)
            ax.fill_between(
                mass_centers[valid_cen],
                (cen_frac_mean - cen_frac_std)[valid_cen],
                (cen_frac_mean + cen_frac_std)[valid_cen],
                color=colors[i], alpha=0.18, linewidth=0)

            # Plot satellites
            ax.plot(mass_centers[valid_sat], sat_frac_mean[valid_sat], color=colors[i], linestyle=':', linewidth=2, alpha=0.25)

        ax.set_xlim(8.0, 12.5)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel(r'$\log_{10} M_* [M_\odot]$', fontsize=14)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        leg = ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
        for text in leg.get_texts():
            text.set_fontsize(8)

    axes[0].set_ylabel('Quenched Fraction', fontsize=14)

    plt.tight_layout()
    save_figure('QuenchedFraction.StellarMassFunctionEvolution', fig, dpi=300, bbox_inches='tight')


def history_halo_mass_function_evolution(models):
    print('Plotting halo mass function evolution')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Define redshift bins (same as stellar mass function evolution)
    z_bins = HISTORY_Z_BINS

    # Define colormap - from dark to light/orange (same as SMF)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_bins)))

    # Mass bins for HMF
    mass_bins = np.arange(10.0, 15.0, 0.1)
    mass_centers = mass_bins[:-1] + 0.05
    bin_width = mass_bins[1] - mass_bins[0]

    for k, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            # Calculate halo mass function for each snapshot separately
            phi_snapshots = []
            for snap_idx in snap_indices:
                halo_mass = m.get(snap_idx, 'Mvir')
                # Get central galaxies only (unique halos)
                central_mask = (m.get(snap_idx, 'Type') == 0) & (halo_mass > 0)
                halo_masses = halo_mass[central_mask]

                if len(halo_masses) > 0:
                    counts, _ = np.histogram(np.log10(halo_masses), bins=mass_bins)
                    phi_snapshots.append(counts / (m.volume * bin_width))

            if len(phi_snapshots) == 0:
                continue

            phi_snapshots = np.array(phi_snapshots)

            # Calculate mean and 1-sigma error (standard error of the mean)
            phi = np.mean(phi_snapshots, axis=0)
            phi_err = np.std(phi_snapshots, axis=0) / np.sqrt(len(phi_snapshots))

            # Only plot where we have data
            valid = (phi > 0) & (phi_err > 0)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if k == 0 else None
            ax.plot(mass_centers[valid], np.log10(phi[valid]),
                    color=colors[i], linewidth=2, ls=m.ls, label=label)
            ax.fill_between(mass_centers[valid],
                           np.log10(phi[valid] - phi_err[valid]),
                           np.log10(phi[valid] + phi_err[valid]),
                           color=colors[i], alpha=0.3)

    ax.set_xlim(10.0, 15.0)
    ax.set_ylim(-6, -0.5)

    ax.set_xlabel(r'$\log_{10} M_{\rm vir}$ [$M_\odot$]', fontsize=14)
    ax.set_ylabel(r'$\log_{10}\phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    leg = ax.legend(loc='upper right', fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontsize(10)
    history_add_model_legend(ax, models, loc='lower left')

    plt.tight_layout()
    save_figure('HaloMassFunctionEvolution', dpi=300, bbox_inches='tight')


def history_concentration_evolution(models):
    print('Plotting median concentration vs redshift by mass bin')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    mass_bins = [
        (1e10, 1e11, r'$10^{10} - 10^{11}$'),
        (1e11, 1e12, r'$10^{11} - 10^{12}$'),
        (1e12, 1e13, r'$10^{12} - 10^{13}$'),
        (1e13, 1e15, r'$10^{13} - 10^{15}$'),
    ]
    colors_cm = ['dodgerblue', 'forestgreen', 'darkorange', 'firebrick']

    for k, m in enumerate(models):
        z_arr = []
        medians = {i: [] for i in range(len(mass_bins))}
        p16 = {i: [] for i in range(len(mass_bins))}
        p84 = {i: [] for i in range(len(mass_bins))}

        for snap in m.history_snaps:
            conc = m.get(snap, 'Concentration')
            mvir = m.get(snap, 'Mvir')
            if len(conc) == 0 or len(mvir) == 0:
                continue

            z_arr.append(m.redshifts[snap])

            for i, (mlo, mhi, _) in enumerate(mass_bins):
                w = np.where((conc > 0) & (mvir >= mlo) & (mvir < mhi))[0]
                if len(w) >= 5:
                    medians[i].append(np.median(conc[w]))
                    p16[i].append(np.percentile(conc[w], 16))
                    p84[i].append(np.percentile(conc[w], 84))
                else:
                    medians[i].append(np.nan)
                    p16[i].append(np.nan)
                    p84[i].append(np.nan)

        z_arr = np.array(z_arr)
        for i, (_, _, label) in enumerate(mass_bins):
            med = np.array(medians[i])
            lo = np.array(p16[i])
            hi = np.array(p84[i])
            valid = np.isfinite(med)
            if np.any(valid):
                ax.plot(z_arr[valid], med[valid], color=colors_cm[i], lw=2.5, ls=m.ls,
                        label=label + r' $M_\odot$' if k == 0 else None)
                ax.fill_between(z_arr[valid], lo[valid], hi[valid],
                                color=colors_cm[i], alpha=0.15)

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'Concentration $c$', fontsize=14)
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    history_add_model_legend(ax, models, loc='upper left', fontsize=12)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    save_figure('ConcentrationEvolution', dpi=300, bbox_inches='tight')


def history_metallicity_evolution(models):
    print('Plotting metallicity vs stellar mass at different redshifts')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS
    colors_z = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_bins)))

    mass_bins = np.arange(7.2, 12.5, 0.1)
    mass_centers = mass_bins[:-1] + 0.05

    for k, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            metallicity_snapshots = []
            for snap_idx in snap_indices:
                stellar_mass = m.get(snap_idx, 'StellarMass')
                # The original plots log10 of MetalsStellarMass as stored (1e10 Msun/h), so
                # undo the Msun conversion that Model.get applies to it
                metals = m.get(snap_idx, 'MetalsStellarMass') * m.Hubble_h / m.UnitMass_Msun
                w = np.where((stellar_mass > 0.0) & (metals > 0.0))[0]
                if len(w) > 0:
                    masses = np.log10(stellar_mass[w])
                    metallicities = np.log10(metals[w])
                    metallicity_bin_means = np.full(len(mass_centers), np.nan)
                    for j in range(len(mass_centers)):
                        mask = (masses >= mass_bins[j]) & (masses < mass_bins[j+1])
                        if np.sum(mask) > 0:
                            metallicity_bin_means[j] = np.mean(metallicities[mask])
                    metallicity_snapshots.append(metallicity_bin_means)

            if len(metallicity_snapshots) == 0:
                continue

            metallicity_means_stack = np.array(metallicity_snapshots)
            metallicity_mean = np.nanmean(metallicity_means_stack, axis=0)
            metallicity_err = np.nanstd(metallicity_means_stack, axis=0) / np.sqrt(len(metallicity_means_stack))
            valid = ~np.isnan(metallicity_mean) & ~np.isnan(metallicity_err)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if k == 0 else None
            ax.plot(mass_centers[valid], metallicity_mean[valid], color=colors_z[i], linewidth=2, ls=m.ls, label=label)
            ax.fill_between(mass_centers[valid],
                           metallicity_mean[valid] - metallicity_err[valid],
                           metallicity_mean[valid] + metallicity_err[valid],
                           color=colors_z[i], alpha=0.3)

    ax.set_xlabel(r'$\log_{10} M_* [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\log_{10} Z$', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    leg = ax.legend(loc='upper left', fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontsize(10)
    history_add_model_legend(ax, models)

    plt.tight_layout()
    save_figure('MetallicityEvolution', dpi=300, bbox_inches='tight')


def history_h2_fraction_evolution(models):
    print('Plotting H2 fraction vs stellar mass at different redshifts')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    z_bins = HISTORY_Z_BINS
    colors_z = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_bins)))

    HYDROGEN_MASS_FRAC = 0.74
    mass_bins = np.arange(7.2, 12.5, 0.1)
    mass_centers = mass_bins[:-1] + 0.05

    for k, m in enumerate(models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            h2_fraction_snapshots = []
            for snap_idx in snap_indices:
                mstar = m.get(snap_idx, 'StellarMass')
                h2 = m.get(snap_idx, 'H2gas')
                cold = m.get(snap_idx, 'ColdGas')
                if not (len(mstar) == len(cold) == len(h2) > 0):
                    continue
                w = np.where((mstar > 0.0) & (cold > 0.0))[0]
                if len(w) == 0:
                    continue
                # f_H2 = H2 / (ColdGas * 0.74): fraction of hydrogen that is molecular.
                # Clip to [0,1]: H2gas can exceed ColdGas in the output because it is
                # computed before SF depletes ColdGas and is not updated afterwards.
                h2_frac_vals = np.clip(h2[w] / (cold[w] * HYDROGEN_MASS_FRAC), 0.0, 1.0)
                masses = np.log10(mstar[w])
                h2_frac_bin_means = np.full(len(mass_centers), np.nan)
                for j in range(len(mass_centers)):
                    mask = (masses >= mass_bins[j]) & (masses < mass_bins[j+1])
                    if np.sum(mask) > 0:
                        h2_frac_bin_means[j] = np.mean(h2_frac_vals[mask])
                h2_fraction_snapshots.append(h2_frac_bin_means)

            if len(h2_fraction_snapshots) == 0:
                continue

            h2_frac_means_stack = np.array(h2_fraction_snapshots)
            h2_frac_mean = np.nanmean(h2_frac_means_stack, axis=0)
            h2_frac_err = np.nanstd(h2_frac_means_stack, axis=0) / np.sqrt(len(h2_frac_means_stack))
            valid = ~np.isnan(h2_frac_mean) & ~np.isnan(h2_frac_err)
            if not np.any(valid):
                continue

            label = f'{z_min:.1f} < z < {z_max:.1f}' if k == 0 else None
            ax.plot(mass_centers[valid], h2_frac_mean[valid], color=colors_z[i], linewidth=2, ls=m.ls, label=label)
            ax.fill_between(mass_centers[valid],
                            h2_frac_mean[valid] - h2_frac_err[valid],
                            h2_frac_mean[valid] + h2_frac_err[valid],
                            color=colors_z[i], alpha=0.3)

    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlim(7.2, 12.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r'$\log_{10} M_* [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$f_{\rm H_2} = M_{\rm H_2}\,/\,(M_{\rm cold} \times 0.74)$', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    leg = ax.legend(loc='upper left', fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontsize(10)
    history_add_model_legend(ax, models)

    plt.tight_layout()
    save_figure('H2FractionEvolution', dpi=300, bbox_inches='tight')


def history_z_to_age_gyr(z, OmegaM, OmegaL, Hubble_h):
    # Numerical integration of dt/dz for flat LCDM; 1/H0 in Gyr = 9.778/h
    z_int = np.linspace(z, 1000.0, 10000)
    E_z = np.sqrt(OmegaM * (1 + z_int)**3 + OmegaL)
    return (9.778 / Hubble_h) * np.trapezoid(1.0 / ((1 + z_int) * E_z), z_int)


def history_star_formation_efficiency(models):
    print('Plotting star formation efficiency across cosmic time')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    t_today = 0.0
    for m in models:
        f_baryon = m.BaryonFrac  # Cosmological baryon fraction Omega_b / Omega_m
        OmegaM = m.Omega
        OmegaL = m.OmegaLambda

        t_arr_sfe = []
        med_sfe = []
        p16_sfe = []
        p84_sfe = []

        for snap in m.history_snaps:
            mstar = m.get(snap, 'StellarMass')
            mvir = m.get(snap, 'Mvir')
            gtype = m.get(snap, 'Type')
            if len(mstar) == 0 or len(mvir) == 0:
                continue

            w = np.where((gtype == 0) & (mstar > 0) & (mvir > 0))[0]
            if len(w) < 5:
                continue

            sfe = mstar[w] / (mvir[w] * f_baryon)
            t_arr_sfe.append(history_z_to_age_gyr(m.redshifts[snap], OmegaM, OmegaL, m.Hubble_h))
            med_sfe.append(np.median(sfe))
            p16_sfe.append(np.percentile(sfe, 16))
            p84_sfe.append(np.percentile(sfe, 84))

        # Sort by ascending cosmic time so the line and fill render correctly
        sort_idx = np.argsort(t_arr_sfe)
        t_arr_sfe = np.array(t_arr_sfe)[sort_idx]
        med_sfe = np.array(med_sfe)[sort_idx]
        p16_sfe = np.array(p16_sfe)[sort_idx]
        p84_sfe = np.array(p84_sfe)[sort_idx]

        t_today = max(t_today, history_z_to_age_gyr(0.0, OmegaM, OmegaL, m.Hubble_h))

        ax.plot(t_arr_sfe, med_sfe, color=m.color, lw=2.5, label=m.label)
        ax.fill_between(t_arr_sfe, p16_sfe, p84_sfe, color=m.color, alpha=0.3)

    ax.set_yscale('log')
    ax.set_xlim(t_today, 0.3)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel(r'Cosmic time [Gyr]', fontsize=14)
    ax.set_ylabel(r'$\varepsilon_* = M_* \,/\, (f_b \cdot M_{\rm vir})$', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.legend(loc='lower left', fontsize=12, frameon=False)

    plt.tight_layout()
    save_figure('StarFormationEfficiency', dpi=300, bbox_inches='tight')


def history_central_satellite_counts(models):
    print('Plotting histogram of Centrals with satellites overlaid as a function of redshift')

    z_bins = HISTORY_Z_BINS
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(z_bins)))

    # Bar colour already means redshift bin, so each model gets a panel
    fig, axes = model_panels(models, ncols_max=2, panel_size=(10, 7), sharey=True)

    for ax, m in zip(axes, models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            central_counts = []
            satellite_counts = []
            for snap_idx in snap_indices:
                types = m.get(snap_idx, 'Type')
                central_counts.append(np.sum(types == 0))
                satellite_counts.append(np.sum(types >= 1))  # >= 1: orphans (Type 2) are satellites too

            central_mean = np.mean(central_counts)
            satellite_mean = np.mean(satellite_counts)
            label = f'{z_min:.1f} < z < {z_max:.1f}'
            ax.bar(i - 0.15, central_mean, width=0.3, color=colors[i], label=label + ' (Centrals)')
            ax.bar(i + 0.15, satellite_mean, width=0.3, color=colors[i], alpha=0.5, label=label + ' (Satellites)')

        ax.set_xlim(-0.5, len(z_bins) - 0.5)
        ax.set_xticks(range(len(z_bins)))
        ax.set_xticklabels([f'{z_min:.1f}-{z_max:.1f}' for z_min, z_max in z_bins], rotation=45, ha='right')

    axes[0].set_ylim(0, 1.1 * max(axes[0].get_ylim()))
    axes[0].set_ylabel('Average Count', fontsize=14)

    plt.tight_layout()
    save_figure('CentralSatelliteCounts', fig, dpi=300, bbox_inches='tight')


def history_central_satellite_fraction(models):
    print('Plotting central/satellite fraction as a function of redshift')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    z_max_all = 0.0
    for k, m in enumerate(models):
        z_arr_cs = []
        central_frac = []
        satellite_frac = []

        for snap in m.history_snaps:
            types = m.get(snap, 'Type')
            n_total = len(types)
            if n_total == 0:
                continue
            n_cen = np.sum(types == 0)
            n_sat = np.sum(types >= 1)  # >= 1: orphans (Type 2) are satellites too
            z_arr_cs.append(m.redshifts[snap])
            central_frac.append(n_cen / n_total)
            satellite_frac.append(n_sat / n_total)

        sort_idx = np.argsort(z_arr_cs)
        z_arr_cs = np.array(z_arr_cs)[sort_idx]
        central_frac = np.array(central_frac)[sort_idx]
        satellite_frac = np.array(satellite_frac)[sort_idx]
        if len(z_arr_cs) > 0:
            z_max_all = max(z_max_all, z_arr_cs.max())

        ax.plot(z_arr_cs, central_frac, color='steelblue', lw=2.5, ls=m.ls, label='Centrals' if k == 0 else None)
        ax.plot(z_arr_cs, satellite_frac, color='firebrick', lw=2.5, ls=m.ls, label='Satellites' if k == 0 else None)

    ax.set_xlim(0, z_max_all if z_max_all > 0 else 12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel('Fraction', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.legend(loc='center right', fontsize=12, frameon=False)
    history_add_model_legend(ax, models, loc='upper right', fontsize=12)

    plt.tight_layout()
    save_figure('CentralSatelliteFraction', dpi=300, bbox_inches='tight')


def history_central_satellite_fraction_massive_quiescent(models):
    print('Plotting central/satellite fraction of most massive quiescent galaxies vs redshift')

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)

    z_max_all = 0.0
    for k, m in enumerate(models):
        OmegaM_cs = m.Omega
        OmegaL_cs = m.OmegaLambda
        inv_H0_yr = (9.778 / m.Hubble_h) * 1.0e9  # 1/H0 in years

        z_arr_mq = []
        central_frac_mq = []
        satellite_frac_mq = []

        for snap in m.history_snaps:
            mstar = m.get(snap, 'StellarMass')
            types = m.get(snap, 'Type')
            sfr_total = m.get(snap, 'SfrDisk') + m.get(snap, 'SfrBulge')
            if len(mstar) == 0:
                continue

            w = np.where(mstar > 0.0)[0]
            if len(w) < 10:
                continue

            # Top 10% by stellar mass at this snapshot
            mass_thresh = np.percentile(mstar[w], 90)
            massive = w[mstar[w] >= mass_thresh]
            if len(massive) == 0:
                continue

            # sSFR threshold = 0.2 / t_Hubble(z) in 1/yr
            z = m.redshifts[snap]
            E_z = np.sqrt(OmegaM_cs * (1.0 + z)**3 + OmegaL_cs)
            ssfr_thresh = 0.2 * E_z / inv_H0_yr

            sSFR = sfr_total[massive] / mstar[massive]
            quiescent = massive[sSFR < ssfr_thresh]
            n_q = len(quiescent)
            if n_q == 0:
                continue

            n_cen = np.sum(types[quiescent] == 0)
            n_sat = np.sum(types[quiescent] >= 1)  # >= 1: orphans (Type 2) are satellites too

            z_arr_mq.append(z)
            central_frac_mq.append(n_cen / n_q)
            satellite_frac_mq.append(n_sat / n_q)

        sort_idx = np.argsort(z_arr_mq)
        z_arr_mq = np.array(z_arr_mq)[sort_idx]
        central_frac_mq = np.array(central_frac_mq)[sort_idx]
        satellite_frac_mq = np.array(satellite_frac_mq)[sort_idx]
        if len(z_arr_mq) > 0:
            z_max_all = max(z_max_all, z_arr_mq.max())

        ax.plot(z_arr_mq, central_frac_mq, color='steelblue', lw=2.5, ls=m.ls, label='Centrals' if k == 0 else None)
        ax.plot(z_arr_mq, satellite_frac_mq, color='firebrick', lw=2.5, ls=m.ls, label='Satellites' if k == 0 else None)

    ax.set_xlim(0, z_max_all if z_max_all > 0 else 12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel('Fraction (top 10% $M_*$, sSFR $< 0.2/t_{\\rm H}$)', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.legend(loc='center right', fontsize=12, frameon=False)
    history_add_model_legend(ax, models, loc='upper right', fontsize=12)

    plt.tight_layout()
    save_figure('CentralSatelliteFraction_MassiveQuiescent', dpi=300, bbox_inches='tight')


def history_quenched_central_satellite_counts(models):
    print('Plotting histogram of quiecent centrals with satellites overlaid as a function of redshift')

    z_bins = HISTORY_Z_BINS
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(z_bins)))

    # Bar colour already means redshift bin, so each model gets a panel
    fig, axes = model_panels(models, ncols_max=2, panel_size=(10, 7), sharey=True)

    for ax, m in zip(axes, models):
        for i, (z_min, z_max) in enumerate(z_bins):
            snap_indices = history_snaps_in_z_range(m, z_min, z_max)
            if len(snap_indices) == 0:
                continue

            quenched_central_counts = []
            quenched_satellite_counts = []
            for snap_idx in snap_indices:
                types = m.get(snap_idx, 'Type')
                sfr_total = m.get(snap_idx, 'SfrDisk') + m.get(snap_idx, 'SfrBulge')
                stellar_mass = m.get(snap_idx, 'StellarMass')
                sSFR_linear = sfr_total / stellar_mass
                quenched = sSFR_linear < 10.0**sSFRcut
                quenched_central_counts.append(np.sum((types == 0) & quenched))
                quenched_satellite_counts.append(np.sum((types >= 1) & quenched))  # >= 1: orphans too

            central_mean = np.mean(quenched_central_counts)
            satellite_mean = np.mean(quenched_satellite_counts)
            label = f'{z_min:.1f} < z < {z_max:.1f}'
            ax.bar(i - 0.15, central_mean, width=0.3, color=colors[i], label=label + ' (Quenched Centrals)')
            ax.bar(i + 0.15, satellite_mean, width=0.3, color=colors[i], alpha=0.5, label=label + ' (Quenched Satellites)')

        ax.set_xlim(-0.5, len(z_bins) - 0.5)
        ax.set_xticks(range(len(z_bins)))
        ax.set_xticklabels([f'{z_min:.1f}-{z_max:.1f}' for z_min, z_max in z_bins], rotation=45, ha='right')

    axes[0].set_ylim(0, 1.1 * max(axes[0].get_ylim()))
    axes[0].set_ylabel('Average Count', fontsize=14)

    plt.tight_layout()
    save_figure('QuenchedCentralSatelliteCounts', fig, dpi=300, bbox_inches='tight')


HISTORY_PLOTS_B = [
    history_quenched_fraction_evolution,
    history_halo_mass_function_evolution,
    history_concentration_evolution,
    history_metallicity_evolution,
    history_h2_fraction_evolution,
    history_star_formation_efficiency,
    history_central_satellite_counts,
    history_central_satellite_fraction,
    history_central_satellite_fraction_massive_quiescent,
    history_quenched_central_satellite_counts,
]


LOCAL_PLOTS = LOCAL_PLOTS_A + LOCAL_PLOTS_B
HISTORY_PLOTS = HISTORY_PLOTS_A + HISTORY_PLOTS_B

# ==================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot SAGE26 local and history diagnostics for the models in MODELS')
    parser.add_argument('--local', action='store_true', help='Only the single-snapshot plots')
    parser.add_argument('--history', action='store_true', help='Only the redshift-evolution plots')
    parser.add_argument('--only', nargs='+', default=None, metavar='NAME',
                        help='Only the plot functions whose names contain one of these strings')
    return parser.parse_args()


if __name__ == '__main__':

    print('Running allresults (compare)\n')
    args = parse_arguments()

    OutputDir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.join(TOP_DIR, OUTPUT_DIR)
    if not OutputDir.endswith(os.sep):
        OutputDir += os.sep
    os.makedirs(OutputDir, exist_ok=True)
    DataDir = DATA_DIR if DATA_DIR is not None else os.path.join(SCRIPT_DIR, '..', 'data')

    print(f'Cuts: Len >= {PARTICLE_LEN_CUT}, Mvir >= {MVIR_CUT:.3g} Msun, '
          f'sSFR cut = {sSFRcut}')
    print('Models:')
    colors = model_colors(len(MODELS))
    models = [Model(spec, colors[i], MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)])
              for i, spec in enumerate(MODELS)]
    print(f'Output directory: {OutputDir}\n')

    run_local = args.local or not args.history
    run_history = args.history or not args.local
    plots = (LOCAL_PLOTS if run_local else []) + (HISTORY_PLOTS if run_history else [])
    if args.only:
        plots = [p for p in plots if any(s in p.__name__ for s in args.only)]

    for plot in plots:
        run_plot(plot, models)

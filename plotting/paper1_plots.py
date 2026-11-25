#!/usr/bin/env python
"""
paper1_plots.py -- SAGE26 Paper I: every figure made from the model runs, plus the
referee-response figures, and the numbers the text quotes.

Figures (file names as written by the functions named):
  2   SFRD history                     paper_plots.plot_16_sfrd_history
  3   Mass-metallicity relation        paper_plots.plot_3_gas_metallicity_vs_stellar_mass
  4   Cold / HI / H2 MFs + SMF         paper1_referee.task_gas            (gas_fig4)
  5   SFR vs stellar mass              paper_plots.plot_15_sfr_vs_stellar_mass
  6   Mass loading and ejection        paper_plots.plot_24_mass_loading_vs_velocity
  7   Stellar-halo mass relation       paper_plots.plot_5_stellar_halo_mass
  8   SMF, star-forming / quiescent    paper_plots.plot_1_stellar_mass_function_ssfr_combined
  9   Black hole-bulge mass            paper_plots.plot_4_bh_bulge_mass
  10  Bulge size-mass by channel       paper_plots.plot_6b_bulge_mass_size_median
  11  FFB fraction vs halo mass        paper_plots.plot_13_ffb_vs_redshift
  12  SMF grid, z = 0-12               paper_plots.plot_18_smf_redshift_grid
  13  Ablation series                  ablation_series.py (mini-Millennium variant runs; --ablation)
  R   Referee letter: cosmology (Fig 1), reionisation, SMF and SFRD offsets
                                       paper1_referee.task_cosmo / reion / smf / sfrd
(Figure 1 is a schematic.)

The plotting code is not copied: the functions above draw the figures, so the
figures and their names are exactly those of paper_plots.py and paper1_referee.py.
What changes is how they get their data, which is what makes the full boxes slow:
  * One column cache.  Every snapshot column is read once, from all MPI files in
    parallel, Len >= 20 applied, and shared by every figure.  The z = 0 catalogues
    and the multi-snapshot data are loaded lazily, column by column, so only
    what a figure touches is read.  After each figure every snapshot but z = 0 is
    freed, so memory stays at about one figure's worth of columns.
  * Mass-function bootstraps are drawn from the binned counts (multinomial),
    which has the same distribution as resampling the galaxies, instead of
    re-histogramming 100 resampled copies of up to ~10^7 galaxies.
  * Bootstrapped means of more than 10^5 values (Fig 11) use their normal limit,
    as paper_plots already does for the summed densities (BOOT_EXACT_MAX).
Everything printed is also written to <out>/paper1_numbers.txt.

Usage (from anywhere; paths are relative to the SAGE26 root):
    python plotting/paper1_plots.py                    # mini-Millennium runs in output/
    python plotting/paper1_plots.py --box big -j 32    # full boxes (Millennium_*, miniUchuu_FFB) in output/
    python plotting/paper1_plots.py --figs 2 12 R      # some figures only
"""

import os
import sys
import time
import argparse
import importlib.util
import subprocess
from collections.abc import Mapping

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import numpy as np
import h5py as h5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FIGS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'R']


# ========================== PARALLEL COLUMN CACHE ==========================

def _read_file(task):
    """Worker: raw columns *props* of group *key* in one file, or None if the group is absent."""
    path, key, props = task
    with h5.File(path, 'r') as f:
        if key not in f:
            return None
        g = f[key]
        return {p: g[p][()] for p in props if p in g}


class ColumnCache:
    """(files, snapshot, column) -> concatenated raw array, read once, in parallel over files."""

    def __init__(self, workers):
        self.workers, self.pool = workers, None
        self.raw, self.found, self.view = {}, {}, {}
        self.n_read = 0

    def _map(self, tasks):
        if self.workers <= 1 or len(tasks) <= 1:
            return [_read_file(t) for t in tasks]
        if self.pool is None:
            import multiprocessing as mp
            ctx = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
            self.pool = ctx.Pool(self.workers)
        return self.pool.map(_read_file, tasks, chunksize=1)

    def columns(self, files, key, props, mass_convert, min_len, mass_props):
        files = tuple(files)
        want = list(dict.fromkeys(list(props) + ['Len']))
        missing = [p for p in want if (files, key, p) not in self.raw]
        if missing:
            res = self._map([(fp, key, missing) for fp in files])
            self.found[(files, key)] = self.found.get((files, key), False) or any(r is not None for r in res)
            for p in missing:
                ch = [r[p] for r in res if r is not None and p in r]
                self.raw[(files, key, p)] = np.concatenate(ch) if ch else None
                self.n_read += 1
        if not self.found.get((files, key)):
            return {}
        out = {}
        for p in want:
            vk = (files, key, p, mass_convert, min_len)
            if vk not in self.view:
                a = self.raw[(files, key, p)]
                if a is None:
                    continue
                ln = self.raw[(files, key, 'Len')]
                if ln is not None and min_len > 0:
                    a = a[ln >= min_len]
                if p in mass_props:
                    a = a * mass_convert
                a.flags.writeable = False          # shared between figures: never modified in place
                self.view[vk] = a
            out[p] = self.view[vk]
        if 'Len' not in props:
            out.pop('Len', None)
        return out

    def trim(self, keep_key):
        """Free every snapshot except *keep_key* (the z = 0 catalogues the next figures share)."""
        for d in (self.raw, self.view, self.found):
            for k in [k for k in d if k[1] != keep_key]:
                del d[k]

    def fields(self, files, key):
        with h5.File(files[0], 'r') as f:
            return list(f[key].keys()) if key in f else []

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


class LazySnap(Mapping):
    """A snapshot catalogue whose columns are read from the cache when first used."""

    def __init__(self, pp, cache, directory, snap):
        self.pp, self.cache = pp, cache
        self.files = pp.find_model_files(directory)
        self.key = f'Snap_{snap}'
        self._fields = None

    def _f(self):
        if self._fields is None:
            self._fields = [p for p in self.cache.fields(self.files, self.key) if p != 'Len'] if self.files else []
        return self._fields

    def __getitem__(self, p):
        d = self.pp.read_snap_from_files(self.files, self.key, [p])
        if p not in d:
            raise KeyError(p)
        return d[p]

    def __contains__(self, p):
        return p in self._f()

    def __iter__(self):
        return iter(self._f())

    def __len__(self):
        return len(self._f())


class LazySnapdata(Mapping):
    """{snap: LazySnap} over the snapshots present in a run."""

    def __init__(self, pp, cache, directory):
        self.pp, self.cache, self.directory = pp, cache, directory
        files = pp.find_model_files(directory)
        with h5.File(files[0], 'r') as f:
            self.snaps = sorted(int(k.split('_')[1]) for k in f if k.startswith('Snap_'))

    def __getitem__(self, s):
        if s not in self.snaps:
            raise KeyError(s)
        return LazySnap(self.pp, self.cache, self.directory, s)

    def __contains__(self, s):
        return s in self.snaps

    def __iter__(self):
        return iter(self.snaps)

    def __len__(self):
        return len(self.snaps)


# ========================== FAST STATISTICS ==========================

def install_fast_paths(pp, cache):
    """Route paper_plots' readers through the cache and swap in the count-based bootstraps."""
    orig_read_snap = pp.read_snap

    def read_snap_from_files(filepaths, snap_key, properties, mass_convert=None):
        mc = pp.MASS_CONVERT if mass_convert is None else mass_convert
        return dict(cache.columns(filepaths, snap_key, properties, mc, pp.MIN_PARTICLES, pp._MASS_PROPS))

    def read_snap(directory, snap, properties, min_particles=None):
        mp_ = pp.MIN_PARTICLES if min_particles is None else min_particles
        files = pp._find_model_files(directory)
        if not files:
            return {}
        return dict(cache.columns(files, f'Snap_{snap}', properties, pp.MASS_CONVERT, mp_, pp._MASS_PROPS))

    def mass_function_bootstrap(log_masses, volume, binwidth=0.1, mass_range=None, n_boot=100):
        centers, phi, mrange = pp.mass_function(log_masses, volume, binwidth, mass_range)
        n = len(log_masses)
        if n == 0:
            return centers, phi, phi, phi, mrange
        mi, ma = mrange
        counts, _ = np.histogram(log_masses, range=(mi, ma), bins=int(round((ma - mi) / binwidth)))
        p = np.append(counts, n - counts.sum()) / n            # last bucket: outside the range
        boots = np.random.multinomial(n, p, size=n_boot)[:, :-1].astype(float)
        with np.errstate(divide='ignore'):
            bp = np.log10(boots / volume / binwidth)
        bp[~np.isfinite(bp)] = np.nan
        with np.errstate(invalid='ignore'):
            return centers, phi, np.nanpercentile(bp, 16, axis=0), np.nanpercentile(bp, 84, axis=0), mrange

    class _CI:
        def __init__(self, lo, hi):
            self.low, self.high = lo, hi

    class _Res:
        def __init__(self, lo, hi):
            self.confidence_interval = _CI(lo, hi)

    class StatsProxy:
        """scipy.stats, except that bootstrapped means of large samples use the normal limit."""

        def __getattr__(self, k):
            return getattr(_scipy_stats, k)

        def bootstrap(self, data, statistic, confidence_level=0.95, **kw):
            x = np.asarray(data[0], dtype=float)
            if statistic is np.mean and x.size > pp.BOOT_EXACT_MAX:
                from scipy.stats import norm
                half = norm.ppf(0.5 + confidence_level / 2) * x.std() / np.sqrt(x.size)
                return _Res(x.mean() - half, x.mean() + half)
            return _scipy_stats.bootstrap(data, statistic, confidence_level=confidence_level, **kw)

    from scipy import stats as _scipy_stats
    pp.read_snap_from_files = read_snap_from_files
    pp.read_snap = read_snap
    pp.mass_function_bootstrap = mass_function_bootstrap
    pp.stats = StatsProxy()
    return orig_read_snap


# ========================== OUTPUT ==========================

class Tee:
    def __init__(self, path):
        self.file, self.stdout = open(path, 'w'), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--box', choices=('mini', 'big'), default='mini',
                    help="run directories: 'mini' (output/millennium*, as paper_plots.py) or 'big' "
                         "(output/Millennium_*, miniUchuu_FFB)")
    ap.add_argument('--figs', nargs='*', default=ALL_FIGS, choices=ALL_FIGS)
    ap.add_argument('--ablation', action='store_true',
                    help='Fig 13: also run ablation_series.py on its existing variant outputs')
    ap.add_argument('-j', '--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    ap.add_argument('--out', default=None, help='output directory (default <primary run>/plots/)')
    args = ap.parse_args()
    if '13' in args.figs and not args.ablation:
        args.figs = [f for f in args.figs if f != '13']

    if args.box == 'big':
        os.environ['SAGE26_RUNS'] = 'big'
    os.chdir(ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'plotting'))
    import matplotlib
    matplotlib.use('Agg')
    import paper_plots as pp

    out = os.path.abspath(args.out) if args.out else os.path.abspath(pp.OUTPUT_DIR)
    os.makedirs(out, exist_ok=True)
    pp.OUTPUT_DIR = out + os.sep
    tee = Tee(os.path.join(out, 'paper1_numbers.txt'))
    sys.stdout = tee

    t0 = time.time()
    cache = ColumnCache(args.workers)
    install_fast_paths(pp, cache)
    pp.setup_style()
    np.random.seed(pp.SEED)
    pp.seed(pp.SEED)
    print(f'Box: {args.box}   primary: {pp.PRIMARY_DIR}   output: {out}   workers: {args.workers}')
    pp.report_imf_audit()

    primary = LazySnap(pp, cache, pp.PRIMARY_DIR, pp.SNAP_Z0)
    vanilla = LazySnap(pp, cache, pp.VANILLA_DIR, pp.SNAP_Z0) if pp.model_files_exist(pp.VANILLA_DIR) else {}

    ref = None
    if '4' in args.figs or 'R' in args.figs:
        ref = load_module('paper1_referee', os.path.join(ROOT, 'plotting', 'random_plotting_scripts',
                                                         'paper1_referee.py'))
        runs = {
            'mill':    ref.try_run(pp.PRIMARY_DIR, 'SAGE26 (Millennium)', ref.COL_MILL, '-'),
            'uchuu':   ref.try_run(pp.MINIUCHUU_DIR, 'SAGE26 (miniUchuu)', ref.COL_UCHUU, '--'),
            'vanilla': ref.try_run(pp.VANILLA_DIR, 'SAGE16', ref.COL_SAGE16, '--'),
            'recipes': {k: ref.try_run(d, k, ref.RECIPE_COLOURS[k]) for k, d in
                        (('GD14', pp.GD14_DIR), ('KD12', pp.KD12_DIR), ('KMT09', pp.KMT09_DIR),
                         ('K13', pp.K13_DIR))},
        }

    jobs = {
        '2':  ('Figure 2', lambda: pp.plot_16_sfrd_history()),
        '3':  ('Figure 3', lambda: pp.plot_3_gas_metallicity_vs_stellar_mass(primary, vanilla)),
        '4':  ('Figure 4', lambda: ref.task_gas(runs, out)),
        '5':  ('Figure 5', lambda: pp.plot_15_sfr_vs_stellar_mass(primary, vanilla)),
        '6':  ('Figure 6', lambda: pp.plot_24_mass_loading_vs_velocity(primary, vanilla)),
        '7':  ('Figure 7', lambda: pp.plot_5_stellar_halo_mass(primary, vanilla)),
        '8':  ('Figure 8', lambda: pp.plot_1_stellar_mass_function_ssfr_combined(primary, vanilla)),
        '9':  ('Figure 9', lambda: pp.plot_4_bh_bulge_mass(primary, vanilla)),
        '10': ('Figure 10', lambda: pp.plot_6b_bulge_mass_size_median(primary, vanilla)),
        '11': ('Figure 11', lambda: pp.plot_13_ffb_vs_redshift(LazySnapdata(pp, cache, pp.PRIMARY_DIR))),
        '12': ('Figure 12', lambda: pp.plot_18_smf_redshift_grid()),
        '13': ('Figure 13', lambda: print(subprocess.run(
            [sys.executable, os.path.join(ROOT, 'plotting', 'ablation_series.py'), '--outdir', out],
            capture_output=True, text=True).stdout)),
        'R':  ('Referee letter', lambda: [ref.TASKS[t](runs, out) for t in ('cosmo', 'reion', 'smf', 'sfrd')]),
    }
    try:
        for f in args.figs:
            title, job = jobs[f]
            t = time.time()
            print(f'\n######## {title} ########')
            job()
            cache.trim(f'Snap_{pp.SNAP_Z0}')
            print(f'  [{title}: {time.time() - t:.1f} s]')
        print(f'\nTotal {time.time() - t0:.1f} s; {cache.n_read} columns read')
    finally:
        cache.close()
        tee.close()
    print(f'Numbers written to {os.path.join(out, "paper1_numbers.txt")}')


if __name__ == '__main__':
    main()

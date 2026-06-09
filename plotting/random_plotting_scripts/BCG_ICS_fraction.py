#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# Hardcoded thresholds for all plots
MVIR_THRESHOLD_REDSHIFT = 1.0e13          # Minimum Mvir in Msun for redshift plots
MVIR_THRESHOLD_REDSHIFT_GROUPS = 10**12.5   # Minimum Mvir in Msun for redshift plots
MVIR_THRESHOLD_MASSFUNCTION = 1.0e10      # Minimum Mvir for mass function plots
MVIR_THRESHOLD_HALOMASS = 1.0e13          # Minimum Mvir for halo mass plots
MVIR_THRESHOLD_FORMATION = 1.0e13         # Minimum Mvir for formation channel plots
STELLARMASS_THRESHOLD = 8.6e8            # Minimum StellarMass in Msun
MIN_SATELLITES = 2                  # Minimum number of satellites to qualify as a group/cluster
BARYON_FRACTION = 0.17              # Cosmic baryon fraction
FORMATION_MASS_FRACTION = 0.5       # Fraction of z=0 mass to define formation time

# Filter for pathological BCGs (undersized centrals from merger tree edge cases)
# BCGs with M_BCG/M_vir < this threshold are flagged as pathological
# Normal BCGs have M_BCG/M_vir ~ 10^-2 to 10^-1; pathological ones have ~ 10^-4 to 10^-5
BCG_MVIR_RATIO_THRESHOLD = 10**-3.5         # M_BCG/M_vir must be > this to be included
FILTER_PATHOLOGICAL_BCGS = True             # Set to False to include all data

# Parallel processing options
NUM_WORKERS = min(8, cpu_count())   # Number of parallel workers (0 = serial)
MIN_FILES_FOR_PARALLEL = 4          # Minimum files to use parallel reading

# Plotting options
OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (12, 18)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 12

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

# Target redshifts for 2x4 grid plots
TARGET_REDSHIFTS = [0, 0.5, 1, 2, 3, 4, 5, 6]

# ==================================================================


def get_script_dir():
    """Get the directory where this script is located"""
    return os.path.dirname(os.path.abspath(__file__))


class DataCache:
    """
    Cache for HDF5 data and computed quantities to avoid redundant reads and calculations.
    """
    def __init__(self, file_list, hubble_h, sim_params):
        self.file_list = file_list
        self.hubble_h = hubble_h
        self.sim_params = sim_params
        self._cache = {}
        self._formation_z_cache = None  # Cache for formation redshifts
        self._satellite_mass_cache = {}  # Cache for satellite mass arrays
        self._satellite_count_cache = {}  # Cache for satellite count arrays

    def get(self, snapshot, fields):
        """
        Get data for a snapshot, reading from cache if available.

        Parameters:
        -----------
        snapshot : str
            Snapshot name (e.g., 'Snap_63')
        fields : list
            List of field names to read

        Returns:
        --------
        dict : Dictionary of field_name -> numpy array
        """
        cache_key = snapshot

        # Check if we have this snapshot cached
        if cache_key not in self._cache:
            self._cache[cache_key] = {}

        # Read any fields not yet cached - use batch reading for efficiency
        result = {}
        fields_to_read = [f for f in fields if f not in self._cache[cache_key]]

        if fields_to_read:
            # Read all missing fields in one pass through files
            data_dict = read_hdf_multi_params(self.file_list, snapshot, fields_to_read)

            # Apply mass conversion and cache each field
            mass_fields = {'Mvir', 'StellarMass', 'IntraClusterStars', 'ColdGas', 'HotGas', 'CGMgas', 'EjectedMass', 'ICS_disrupted', 'ICS_accreted', 'MetalsStellarMass'}
            for field in fields_to_read:
                data = data_dict.get(field, np.array([]))
                # Apply mass conversion for mass fields
                if field in mass_fields and data.size > 0:
                    data = data * 1.0e10 / self.hubble_h
                self._cache[cache_key][field] = data

        # Return requested fields
        for field in fields:
            result[field] = self._cache[cache_key][field]

        return result

    def get_satellite_mass(self, snapshot):
        """
        Get satellite mass summed to centrals (cached).

        Uses numpy searchsorted for fast vectorized lookup instead of Python dicts.
        """
        if snapshot in self._satellite_mass_cache:
            return self._satellite_mass_cache[snapshot]

        data = self.get(snapshot, ['StellarMass', 'Type', 'CentralGalaxyIndex', 'GalaxyIndex'])
        StellarMass = data['StellarMass']
        Type = data['Type']
        CentralGalaxyIndex = data['CentralGalaxyIndex']
        GalaxyIndex = data['GalaxyIndex']

        # Use numpy searchsorted for fast vectorized lookup (much faster than dict)
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]

        satellite_mask = Type != 0
        satellite_central_gids = CentralGalaxyIndex[satellite_mask]
        satellite_masses = StellarMass[satellite_mask]

        # Find positions using searchsorted
        insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
        # Clamp to valid range and check for exact matches
        insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
        valid_match = sorted_gids[insert_pos] == satellite_central_gids

        # Map back to original indices
        central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
        valid_satellites = central_indices >= 0

        # Use np.add.at for fast accumulation
        satellite_mass = np.zeros(len(StellarMass))
        np.add.at(satellite_mass, central_indices[valid_satellites], satellite_masses[valid_satellites])

        self._satellite_mass_cache[snapshot] = satellite_mass
        return satellite_mass

    def get_satellite_count(self, snapshot):
        """
        Get number of satellites per central (cached).
        """
        if snapshot in self._satellite_count_cache:
            return self._satellite_count_cache[snapshot]

        data = self.get(snapshot, ['Type', 'CentralGalaxyIndex', 'GalaxyIndex'])
        Type = data['Type']
        CentralGalaxyIndex = data['CentralGalaxyIndex']
        GalaxyIndex = data['GalaxyIndex']

        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]

        satellite_mask = Type != 0
        satellite_central_gids = CentralGalaxyIndex[satellite_mask]

        insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
        insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
        valid_match = sorted_gids[insert_pos] == satellite_central_gids

        central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)

        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, central_indices[central_indices >= 0], 1)

        self._satellite_count_cache[snapshot] = n_satellites
        return n_satellites

    def get_formation_redshifts(self, snapshot, tree_indices):
        """
        Get formation redshifts for given tree indices (cached).

        Formation redshift = when halo first reached FORMATION_MASS_FRACTION of z=0 mass.
        """
        snap = int(snapshot.replace('Snap_', ''))

        # Check if we have cached formation z for this snapshot
        if self._formation_z_cache is not None and self._formation_z_cache['snap'] == snap:
            cached = self._formation_z_cache
            # Map tree indices to formation redshifts using cached lookup
            tree_to_pos_arr = cached['tree_to_pos_arr']
            tree_formation_z = cached['tree_formation_z']

            # Vectorized lookup
            valid_tree_mask = tree_indices <= len(tree_to_pos_arr) - 1
            formation_z = np.full(len(tree_indices), np.nan)
            valid_indices = tree_indices[valid_tree_mask]
            positions = tree_to_pos_arr[valid_indices]
            valid_pos = positions >= 0
            formation_z[np.where(valid_tree_mask)[0][valid_pos]] = tree_formation_z[positions[valid_pos]]
            return formation_z

        # Compute formation redshifts
        redshifts = self.sim_params['redshifts']
        Hubble_h = self.hubble_h

        # Get z=0 data
        data = self.get(snapshot, ['Mvir', 'Type', 'SAGETreeIndex'])
        Mvir = data['Mvir']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']

        # Get unique trees we need to track
        unique_trees = np.unique(tree_indices)
        n_trees = len(unique_trees)

        # Vectorized z=0 mass calculation
        central_mask_z0 = Type == 0
        tree_idx_centrals_z0 = SAGETreeIndex[central_mask_z0]
        mvir_centrals_z0 = Mvir[central_mask_z0]

        sort_idx_z0 = np.argsort(tree_idx_centrals_z0)
        sorted_trees_z0 = tree_idx_centrals_z0[sort_idx_z0]
        sorted_mvir_z0 = mvir_centrals_z0[sort_idx_z0]

        unique_trees_z0, first_idx_z0 = np.unique(sorted_trees_z0, return_index=True)
        max_mvir_z0 = np.maximum.reduceat(sorted_mvir_z0, first_idx_z0)

        # Use searchsorted for fast lookup of z0 masses
        z0_sort_idx = np.argsort(unique_trees_z0)
        z0_sorted_trees = unique_trees_z0[z0_sort_idx]
        z0_sorted_masses = max_mvir_z0[z0_sort_idx]

        insert_pos = np.searchsorted(z0_sorted_trees, unique_trees)
        insert_pos = np.clip(insert_pos, 0, len(z0_sorted_trees) - 1)
        valid_match = z0_sorted_trees[insert_pos] == unique_trees
        tree_z0_mass = np.where(valid_match, z0_sorted_masses[insert_pos], 0)

        tree_mass_threshold = tree_z0_mass * FORMATION_MASS_FRACTION
        tree_formation_z = np.full(n_trees, np.nan)
        resolved = np.zeros(n_trees, dtype=bool)

        # Create fast lookup array
        tree_to_pos_arr = np.full(unique_trees.max() + 1, -1, dtype=np.int32)
        tree_to_pos_arr[unique_trees] = np.arange(n_trees)

        # Process snapshots from z=0 backward
        first_file = self.file_list[0]
        with h5.File(first_file, 'r') as f:
            for snap_num in range(snap, 0, -1):
                if resolved.all():
                    break

                snap_name = f'Snap_{snap_num}'
                if snap_name not in f:
                    continue

                TreeIdx_snap = np.array(f[snap_name]['SAGETreeIndex'])
                Type_snap = np.array(f[snap_name]['Type'])
                Mvir_snap = np.array(f[snap_name]['Mvir']) * 1.0e10 / Hubble_h

                central_mask = Type_snap == 0
                TreeIdx_centrals = TreeIdx_snap[central_mask]
                Mvir_centrals = Mvir_snap[central_mask]

                if len(TreeIdx_centrals) == 0:
                    continue

                sort_idx = np.argsort(TreeIdx_centrals)
                sorted_trees = TreeIdx_centrals[sort_idx]
                sorted_masses = Mvir_centrals[sort_idx]

                unique_in_snap, first_idx = np.unique(sorted_trees, return_index=True)
                max_masses = np.maximum.reduceat(sorted_masses, first_idx)

                valid_tree_mask = unique_in_snap <= unique_trees.max()
                valid_trees = unique_in_snap[valid_tree_mask]
                valid_max_masses = max_masses[valid_tree_mask]

                if len(valid_trees) == 0:
                    continue

                positions = tree_to_pos_arr[valid_trees]
                in_target = positions >= 0

                target_positions = positions[in_target]
                target_masses = valid_max_masses[in_target]

                unresolved_mask = ~resolved[target_positions]
                below_threshold = target_masses < tree_mass_threshold[target_positions]
                newly_resolved = unresolved_mask & below_threshold

                resolved_positions = target_positions[newly_resolved]
                tree_formation_z[resolved_positions] = redshifts[snap_num]
                resolved[resolved_positions] = True

            tree_formation_z[~resolved] = redshifts[1] if len(redshifts) > 1 else 0

        # Cache the result
        self._formation_z_cache = {
            'snap': snap,
            'tree_to_pos_arr': tree_to_pos_arr,
            'tree_formation_z': tree_formation_z,
            'unique_trees': unique_trees
        }

        # Map back to requested tree indices
        positions = tree_to_pos_arr[tree_indices]
        valid_pos = positions >= 0
        formation_z = np.full(len(tree_indices), np.nan)
        formation_z[valid_pos] = tree_formation_z[positions[valid_pos]]

        return formation_z

    def clear(self):
        """Clear the cache to free memory."""
        self._cache = {}
        self._formation_z_cache = None
        self._satellite_mass_cache = {}

    def clear_snapshot(self, snapshot):
        """Clear a specific snapshot from cache."""
        if snapshot in self._cache:
            del self._cache[snapshot]
        if snapshot in self._satellite_mass_cache:
            del self._satellite_mass_cache[snapshot]


def read_simulation_params(filepath):
    """
    Read simulation parameters from HDF5 file header.
    Returns a dictionary with all relevant parameters.
    """
    params = {}

    with h5.File(filepath, 'r') as f:
        # Read from Header/Simulation
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['Omega'] = float(sim.attrs['omega_matter'])
        params['OmegaLambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = float(sim.attrs['particle_mass'])

        # Read from Header/Runtime
        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])

        # Read snapshot info - these are the redshifts for ALL snapshots
        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        # Find available snapshot groups in the file
        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['first_snapshot'] = min(snap_numbers) if snap_numbers else 0
        params['last_snapshot'] = max(snap_numbers) if snap_numbers else 0

    return params


def _read_single_file(args):
    """Worker function to read a single HDF5 file (for parallel processing)."""
    filepath, snap_num, param = args
    try:
        with h5.File(filepath, 'r') as f:
            if snap_num in f and param in f[snap_num]:
                data = np.array(f[snap_num][param])
                if data.size > 0:
                    return data
    except Exception:
        pass
    return None


def _read_single_file_multi_params(args):
    """Worker function to read multiple parameters from a single HDF5 file."""
    filepath, snap_num, params = args
    result = {}
    try:
        with h5.File(filepath, 'r') as f:
            if snap_num in f:
                for param in params:
                    if param in f[snap_num]:
                        data = np.array(f[snap_num][param])
                        if data.size > 0:
                            result[param] = data
    except Exception:
        pass
    return result


def read_hdf(filepaths, snap_num, param):
    """Read and concatenate a parameter from multiple HDF5 files for a given snapshot.

    Uses parallel reading when NUM_WORKERS > 0 and enough files are present.
    """
    # Use parallel reading for multiple files
    if NUM_WORKERS > 0 and len(filepaths) >= MIN_FILES_FOR_PARALLEL:
        args_list = [(fp, snap_num, param) for fp in filepaths]
        with Pool(min(NUM_WORKERS, len(filepaths))) as pool:
            results = pool.map(_read_single_file, args_list)
        data_list = [r for r in results if r is not None]
    else:
        # Serial reading for small number of files
        data_list = []
        for filepath in filepaths:
            with h5.File(filepath, 'r') as f:
                if snap_num in f and param in f[snap_num]:
                    data = np.array(f[snap_num][param])
                    if data.size > 0:
                        data_list.append(data)

    if not data_list:
        return np.array([])

    return np.concatenate(data_list)


def read_hdf_multi_params(filepaths, snap_num, params):
    """Read and concatenate multiple parameters from HDF5 files in parallel.

    More efficient than calling read_hdf multiple times as it opens each file only once.

    Parameters:
    -----------
    filepaths : list
        List of HDF5 file paths
    snap_num : str
        Snapshot name (e.g., 'Snap_63')
    params : list
        List of parameter names to read

    Returns:
    --------
    dict : Dictionary of param_name -> concatenated numpy array
    """
    # Use parallel reading for multiple files
    if NUM_WORKERS > 0 and len(filepaths) >= MIN_FILES_FOR_PARALLEL:
        args_list = [(fp, snap_num, params) for fp in filepaths]
        with Pool(min(NUM_WORKERS, len(filepaths))) as pool:
            results = pool.map(_read_single_file_multi_params, args_list)
    else:
        # Serial reading
        results = []
        for filepath in filepaths:
            result = {}
            with h5.File(filepath, 'r') as f:
                if snap_num in f:
                    for param in params:
                        if param in f[snap_num]:
                            data = np.array(f[snap_num][param])
                            if data.size > 0:
                                result[param] = data
            results.append(result)

    # Concatenate results from all files
    combined = {}
    for param in params:
        data_list = [r[param] for r in results if param in r]
        if data_list:
            combined[param] = np.concatenate(data_list)
        else:
            combined[param] = np.array([])

    return combined


def get_snapshots_for_redshifts(target_redshifts, redshifts, available_snapshots):
    """
    Find the closest available snapshots for a list of target redshifts.

    Parameters:
    -----------
    target_redshifts : list
        List of target redshift values
    redshifts : array
        Array of redshifts indexed by snapshot number
    available_snapshots : list
        List of available snapshot numbers in the data

    Returns:
    --------
    list : Snapshot numbers closest to the target redshifts
    """
    snaps = []
    for z_target in target_redshifts:
        # Find the snapshot with redshift closest to target among available ones
        best_snap = None
        best_diff = np.inf
        for snap in available_snapshots:
            if snap < len(redshifts):
                diff = abs(redshifts[snap] - z_target)
                if diff < best_diff:
                    best_diff = diff
                    best_snap = snap
        if best_snap is not None:
            snaps.append(best_snap)
    return snaps


def plot_ics_observations(ax):
    """
    Add observational ICS/ICL fraction data points and simulation lines to a plot.

    Parameters:
    -----------
    ax : matplotlib axes
        The axes to plot on
    """
    # Observational data (values in % converted to fraction)
    # Pentagons - Burke, Collins, Stott and Hilton - ICL @ z=1, 2012
    redshifts_1 = [0.9468354430379745, 0.8303797468354429, 0.7949367088607594, 0.8075949367088605, 1.2227848101265821]
    ihs_fraction_1 = np.array([1.415094339622641, 2.594339622641499, 3.7735849056603783, 1.5330188679245182, 2.3584905660377373]) / 100

    # Circles (gray) - Montes and Trujillo - ICL at the Frontier, 2018
    redshifts_2 = [0.5341772151898734, 0.5443037974683542, 0.36708860759493667, 0.39746835443037976, 0.3417721518987341,
                   0.30379746835443033, 0.04810126582278479]
    ihs_fraction_2 = np.array([1.5330188679245182, 0, 1.0613207547169807, 1.5330188679245182, 2.7122641509433976,
                               3.3018867924528266, 8.60849056603773]) / 100

    # Squares - Burke, Hilton and Collins, ICL and CLASH, 2015
    redshifts_3 = [0.4025316455696200, 0.3873417721518990, 0.39746835443038000, 0.33924050632911400, 0.3443037974683540,
                   0.3417721518987340, 0.2911392405063290, 0.2253164556962030, 0.2177215189873420, 0.21265822784810100,
                   0.19493670886075900, 0.17721518987341800]
    ihs_fraction_3 = np.array([2.594339622641500, 2.7122641509434000, 3.3018867924528300, 5.542452830188670, 6.014150943396220,
                               7.193396226415100, 12.971698113207500, 12.500000000000000, 16.27358490566040, 18.042452830188700,
                               16.863207547169800, 23.113207547169800]) / 100

    # Crosses - Furnell et al., Growth of ICL in XCS-HSC from 0.1<z<0.5, 2021
    redshifts_4 = [0.1443037974683540, 0.12658227848101300, 0.12151898734177200, 0.08101265822784800, 0.2253164556962030,
                   0.21518987341772100, 0.2556962025316460, 0.3063291139240510, 0.260759493670886, 0.2936708860759490, 0.3215189873417720,
                   0.3417721518987340, 0.3721518987341770, 0.3367088607594940, 0.37721518987341800, 0.3291139240506330, 0.4962025316455700,
                   0.42531645569620200, 0.10886075949367100]
    ihs_fraction_4 = np.array([38.561320754717000, 30.660377358490600, 31.014150943396200, 28.891509433962300, 26.533018867924500,
                               23.58490566037740, 28.5377358490566, 29.716981132075500, 32.54716981132080, 27.476415094339600, 27.594339622641500,
                               26.650943396226400, 19.81132075471700, 18.867924528301900, 15.448113207547200, 15.330188679245300,
                               11.320754716981100, 9.669811320754720, 31.603773584905700]) / 100

    # Down Triangles - Feldmeier et al., Deep CCD, 2004
    redshifts_6 = [0.16202531645569600, 0.16202531645569600, 0.16202531645569600, 0.18481012658227800]
    ihs_fraction_6 = np.array([15.212264150943400, 12.146226415094300, 10.259433962264100, 7.311320754716980]) / 100

    # Black Circles - Montes and Trujillo - ICL at the Frontier, 2018
    redshifts_7 = [0.30126582278481000, 0.38987341772151900, 0.3417721518987340, 0.5367088607594940,
                   0.5367088607594940, 0.36962025316455700, 0.043037974683544300]
    ihs_fraction_7 = np.array([7.665094339622630, 8.60849056603773, 13.089622641509400, 6.603773584905650,
                               5.778301886792450, 4.834905660377360, 10.849056603773600]) / 100

    # Star - Ko and Jee, Existence of ICL at z = 1.24, 2018
    redshifts_8 = [1.2379746835443037]
    ihs_fraction_8 = np.array([9.905660377358487]) / 100

    # Black Diamond - Kluge et al., ICL and host Cluster, 2021
    redshifts_9 = [0.030379746835442978]
    ihs_fraction_9 = np.array([17.924528301886788]) / 100

    # Plus - Zibetti et al., IGS in z=0.25 clusters, 2005
    redshifts_10 = [0.24303797468354427]
    ihs_fraction_10 = np.array([10.849056603773576]) / 100

    # Triangle - Presotto et al., ICL in CLASH-VLT cluster MACS J1206.2-0947, 2014
    redshifts_11 = [0.4354430379746834]
    ihs_fraction_11 = np.array([12.264150943396224]) / 100

    # Black Triangle - Presotto et al., 2014
    redshifts_12 = [0.43291139240506316]
    ihs_fraction_12 = np.array([5.542452830188672]) / 100

    # Black Side Triangle - Spavone et al., Fornax Deep Survey, 2020
    redshifts_13 = [0]
    ihs_fraction_13 = np.array([34.08018867924528]) / 100

    # JWST XLSSC 122 at z=1.98
    redshifts_16 = [1.98]
    ihs_fraction_16 = np.array([17.0]) / 100

    # Ragusa et al. 2023, VEGAS grouops
    redshift_17 = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    ihs_fraction_17 = np.array([0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17, 0.08, 0.35, 0.18, 0.07, 0.20, 0.22, 0.28, 0.30])
    # Ragusa et al. 2023, VEGAS Antlia cluster
    redshifts_18 = [0.05]
    ihs_fraction_18 = np.array([0.35])

    # Ahad et al. 2025 KIDS+GAMA groups
    redshifts_19 = [0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24]
    ihs_fraction_19 = np.array([0.16, 0.10, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15, 0.05]) 

    # Plot observational data (no legend)
    ax.scatter(redshifts_1, ihs_fraction_1, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_2, ihs_fraction_2, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_3, ihs_fraction_3, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_4, ihs_fraction_4, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_6, ihs_fraction_6, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_7, ihs_fraction_7, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_8, ihs_fraction_8, marker='x', color='k', edgecolors='black', s=80, zorder=5, alpha=0.8)
    ax.scatter(redshifts_9, ihs_fraction_9, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_10, ihs_fraction_10, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_11, ihs_fraction_11, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_12, ihs_fraction_12, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_13, ihs_fraction_13, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_16, ihs_fraction_16, marker='*', color='yellow', edgecolors='black', s=100, zorder=5)

    ax.scatter(redshift_17, ihs_fraction_17, marker='+', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_18, ihs_fraction_18, marker='x', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)
    ax.scatter(redshifts_19, ihs_fraction_19, marker='+', color='k', edgecolors='black', s=60, zorder=5, alpha=0.8)

    # Simulation lines (with legend)
    # Rudick, Mihos and McBride, Quantity of ICL, 2011
    redshifts_5 = [0.004011349760203840, 0.023056412555524700, 0.05248969142102060, 0.08192297028651650, 0.12001309587715800,
                   0.1581032214678000, 0.19696284454512100, 0.23774621133914200, 0.2758363369297840, 0.31392646252042500, 0.3546136421286110,
                   0.38880818669293700, 0.42681174381632700, 0.4662869648829930, 0.5043770904736340, 0.542467216064276, 0.5805573416549180,
                   0.6186474672455600, 0.6567375928362010, 0.6948277184268430, 0.7329178440174850, 0.7710079696081270, 0.8090980951987680,
                   0.84718822078941, 0.8852783463800520, 0.9233684719706940, 0.9614585975613350, 0.999548723151977, 1.0341761100525600,
                   1.0711119894131800, 1.1449837481344300, 1.183073873725070, 1.2211639993157100, 1.2592541249063500, 1.2817619263917300, 1.1083123425692700]
    ihs_fraction_5 = np.array([17.035633925701400, 15.255522799283600, 13.375138908022200, 11.340768199977500, 11.708336986061200, 12.328609312577500,
                               12.848617042445400, 12.722826835652300, 11.208443436987300, 11.164335182657300, 11.105524176883900, 11.044875327180100,
                               10.871015291362500, 10.55417099775830, 10.091034327292800, 9.55438389961052, 8.833949078886400, 8.635461934401190, 9.150058234918420,
                               8.554596801462770, 7.782702350686930, 7.51070144898496, 7.701837217748510, 7.643026211975110, 7.326917055943100, 6.790266628260850,
                               6.6579418652707000, 6.7755638768175, 7.223997795839650, 6.9544473527115800, 6.15069694047515, 5.900750165938210,
                               5.966912547433290, 6.047777680371710, 6.011020801763340, 5.882352941176470]) / 100

    # Tang et al., hydrodynamical simulation, 2018
    redshifts_14 = [0.1021897810218980, 0.12408759124087600, 0.13625304136253000,
                    0.15571776155717800, 0.1800486618004870, 0.2068126520681270,
                    0.22871046228710500, 0.25304136253041400, 0.2700729927007300,
                    0.291970802919708, 0.3187347931873480, 0.34063260340632600,
                    0.3673965936739660, 0.3917274939172750, 0.42092457420924600,
                    0.44282238442822400, 0.45985401459854000, 0.4841849148418490]
    ihs_fraction_14 = np.array([22.675736961451200, 21.995464852607700, 20.975056689342400,
                                19.954648526077100, 18.934240362811800, 18.140589569161000,
                                17.006802721088400, 16.099773242630400, 15.532879818594100,
                                14.625850340136000, 13.718820861678000, 13.151927437641700,
                                12.131519274376400, 10.997732426303800, 9.523809523809520,
                                8.73015873015872, 7.596371882086170, 6.68934240362811]) / 100

    # ax.plot(redshifts_5, ihs_fraction_5, linestyle='--', color='plum', lw=2, label='Rudick et al. 2011')
    # ax.plot(redshifts_14, ihs_fraction_14, linestyle='--', color='royaldodgerblue', lw=2, label='Tang et al. 2018')


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Plot SAGE26 BCG and ICS analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "output/millennium/model_*.hdf5"
  %(prog)s output/millennium/model_0.hdf5 --snapshot 63
  %(prog)s "output/millennium/model_*.hdf5" -o my_plots/
        """
    )

    parser.add_argument('input_pattern', nargs='?',
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files (default: ./output/millennium/model_*.hdf5)')

    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number for single-snapshot plots (default: latest available)')

    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots (default: <input_dir>/plots/)')

    return parser.parse_args()


# ==================================================================
# PLOTTING FUNCTIONS
# ==================================================================


def plot_ics_mass_function(file_list, sim_params, output_dir, snaps=None, cache=None):
    """
    Plot ICS Mass Function split by halo mass bins.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    snaps : list, optional
        List of snapshot numbers to plot. If None, uses TARGET_REDSHIFTS.
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Mass Function Plot\n')

    if snaps is None:
        # Get snapshots corresponding to target redshifts
        snaps = get_snapshots_for_redshifts(
            TARGET_REDSHIFTS,
            sim_params['redshifts'],
            sim_params['available_snapshots']
        )

    # Filter to only available snapshots
    snaps = [s for s in snaps if s in sim_params['available_snapshots']]
    if not snaps:
        print('  No valid snapshots available for this plot.')
        return

    Hubble_h = sim_params['Hubble_h']
    BoxSize = sim_params['BoxSize']
    VolumeFraction = sim_params['VolumeFraction']
    redshifts = sim_params['redshifts']

    # Volume for number density calculation
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    # Mass function parameters
    mi_mf = 4.5
    ma_mf = 12.75
    binwidth_mf = 0.25
    NB = int((ma_mf - mi_mf) / binwidth_mf)

    # Halo mass bin edges (in log10 Msun)
    halo_bin_edges = [10.5, 12, 13.5, 17]
    subset_colors = ['firebrick', 'green', 'dodgerblue']
    legend_labels = [r'$10^{10.5} < M_{\mathrm{vir}} < 10^{12}$',
                     r'$10^{12} < M_{\mathrm{vir}} < 10^{13.5}$',
                     r'$M_{\mathrm{vir}} > 10^{13.5}$']

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    # Track statistics
    total_galaxies = 0
    snap_stats = []

    for idx, snap in enumerate(snaps):
        if idx >= len(axes):
            break

        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data from all files (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS, StellarMass, Type = data['Mvir'], data['IntraClusterStars'], data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select central galaxies with valid ICS (applying mass function threshold)
        w = np.where((Type == 0) & (ICS > 0) & (Mvir >= MVIR_THRESHOLD_MASSFUNCTION) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]
        total_galaxies += len(w)
        snap_stats.append((snap, z, len(w)))

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(4.5, 12.0)
            ax.set_ylim(1e-8, 1e-1)
            ax.set_yscale('log')
            ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
            if idx % 2 == 0:
                ax.set_yticklabels(['-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8'])
            else:
                ax.set_yticklabels([])
            continue

        Mvir_sel = Mvir[w]
        ICS_sel = ICS[w]
        log_Mvir = np.log10(Mvir_sel)
        log_ICS = np.log10(ICS_sel)

        # Plot mass function for each halo mass bin
        for bin_idx in range(len(halo_bin_edges) - 1):
            lo = halo_bin_edges[bin_idx]
            hi = halo_bin_edges[bin_idx + 1]
            mask = (log_Mvir >= lo) & (log_Mvir < hi)
            ICS_in_bin = log_ICS[mask]

            if len(ICS_in_bin) > 0:
                counts, binedges = np.histogram(ICS_in_bin, range=(mi_mf, ma_mf), bins=NB)
                xaxeshisto = binedges[:-1] + 0.5 * binwidth_mf
                phi = counts / volume / binwidth_mf
                ax.plot(xaxeshisto, phi, color=subset_colors[bin_idx], alpha=0.7)
                ax.fill_between(xaxeshisto, 0, phi, color=subset_colors[bin_idx], alpha=0.1)

        # Plot overall mass function
        counts, binedges = np.histogram(log_ICS, range=(mi_mf, ma_mf), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth_mf
        phi = counts / volume / binwidth_mf
        ax.plot(xaxeshisto, phi, color='black')

        # Formatting
        ax.set_yscale('log')
        ax.set_xlim(4.5, 12.0)
        ax.set_ylim(1e-8, 1e-1)
        ax.set_xticks([5, 6, 7, 8, 9, 10, 11, 12])

        # Set y-axis ticks to show log values (-1, -2, etc.)
        ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
        if idx % 2 == 0:  # Left column - show tick labels
            ax.set_yticklabels(['-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8'])
        else:  # Right column - hide tick labels
            ax.set_yticklabels([])

        # Add redshift label (upper right)
        ax.text(0.95, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    # Hide any unused axes
    for idx in range(len(snaps), len(axes)):
        axes[idx].set_visible(False)

    # Add x-axis labels at bottom of each column
    axes[6].set_xlabel(r'$\log_{10} m_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14)
    axes[7].set_xlabel(r'$\log_{10} m_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14)

    # Add y-axis labels for each row (left column only)
    ylabel = r'$\log_{10}\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$'
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[2].set_ylabel(ylabel, fontsize=12)
    axes[4].set_ylabel(ylabel, fontsize=12)
    axes[6].set_ylabel(ylabel, fontsize=12)

    # Create legend at bottom of figure
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in subset_colors]
    custom_lines.append(Line2D([0], [0], color='black', lw=2))
    legend_labels_with_threshold = legend_labels + [f'Overall ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_MASSFUNCTION):.1f}}}$ M$_\\odot$)']
    fig.legend(custom_lines, legend_labels_with_threshold,
               loc='lower center', ncol=4, fontsize=10, frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05)

    outputFile = os.path.join(output_dir, 'ICS_mass_function' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps)} (z = {redshifts[snaps[0]]:.2f} to {redshifts[snaps[-1]]:.2f})')
    print(f'  Total centrals with ICS > 0: {total_galaxies}')
    print(f'  Saved: {outputFile}')


def plot_ics_bcg_ratio_distribution(file_list, sim_params, output_dir, snaps=None, cache=None):
    """
    Plot M_ICS/M_BCG ratio distribution across different redshifts in a 2x4 grid.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    snaps : list, optional
        List of snapshot numbers to plot. If None, uses TARGET_REDSHIFTS.
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS/BCG Ratio Distribution Plot\n')

    if snaps is None:
        # Get snapshots corresponding to target redshifts
        snaps = get_snapshots_for_redshifts(
            TARGET_REDSHIFTS,
            sim_params['redshifts'],
            sim_params['available_snapshots']
        )

    # Filter to only available snapshots
    snaps = [s for s in snaps if s in sim_params['available_snapshots']]
    if not snaps:
        print('  No valid snapshots available for this plot.')
        return

    Hubble_h = sim_params['Hubble_h']
    BoxSize = sim_params['BoxSize']
    VolumeFraction = sim_params['VolumeFraction']
    redshifts = sim_params['redshifts']

    # Volume for number density calculation
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    # First pass: collect all ratios and properties to determine dynamic ratio_max
    print('  Calculating ratio statistics...')
    if FILTER_PATHOLOGICAL_BCGS:
        print(f'  Filtering pathological BCGs with M_BCG/M_vir < {BCG_MVIR_RATIO_THRESHOLD:.1e}')

    all_ratios = []
    all_mvir = []
    all_stellar = []
    all_ics = []
    all_z = []
    all_snap = []
    all_galaxyindex = []
    n_pathological_filtered = 0
    n_total_candidates = 0
    for snap in snaps:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'GalaxyIndex'])
            Mvir_tmp, ICS_tmp = data['Mvir'], data['IntraClusterStars']
            StellarMass_tmp, Type_tmp = data['StellarMass'], data['Type']
            GalaxyIndex_tmp = data['GalaxyIndex']
        else:
            Mvir_tmp = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS_tmp = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass_tmp = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type_tmp = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex_tmp = read_hdf(file_list, Snapshot, 'GalaxyIndex')

        # Base selection: Type 0 centrals with ICS and meeting mass thresholds
        base_mask = ((Type_tmp == 0) & (ICS_tmp > 0) &
                     (StellarMass_tmp >= STELLARMASS_THRESHOLD) &
                     (Mvir_tmp >= MVIR_THRESHOLD_MASSFUNCTION))

        n_total_candidates += np.sum(base_mask)

        # Optional filter for pathological BCGs (undersized centrals from merger tree edge cases)
        if FILTER_PATHOLOGICAL_BCGS:
            bcg_mvir_ratio = np.where(Mvir_tmp > 0, StellarMass_tmp / Mvir_tmp, 1.0)
            pathological_mask = base_mask & (bcg_mvir_ratio < BCG_MVIR_RATIO_THRESHOLD)
            n_pathological_filtered += np.sum(pathological_mask)
            valid_mask = base_mask & ~pathological_mask
        else:
            valid_mask = base_mask

        w_tmp = np.where(valid_mask)[0]
        if len(w_tmp) > 0:
            ratios_tmp = ICS_tmp[w_tmp] / StellarMass_tmp[w_tmp]
            all_ratios.extend(ratios_tmp)
            all_mvir.extend(Mvir_tmp[w_tmp])
            all_stellar.extend(StellarMass_tmp[w_tmp])
            all_ics.extend(ICS_tmp[w_tmp])
            all_z.extend([z] * len(w_tmp))
            all_snap.extend([snap] * len(w_tmp))
            all_galaxyindex.extend(GalaxyIndex_tmp[w_tmp])

    # Report pathological BCG filtering
    if FILTER_PATHOLOGICAL_BCGS and n_total_candidates > 0:
        print(f'  Pathological BCGs filtered: {n_pathological_filtered} of {n_total_candidates} '
              f'({100*n_pathological_filtered/n_total_candidates:.2f}%)')

    # Ratio distribution parameters (dynamic ratio_max based on 99th percentile)
    ratio_min = 0.0
    if len(all_ratios) > 0:
        all_ratios = np.array(all_ratios)
        all_mvir = np.array(all_mvir)
        all_stellar = np.array(all_stellar)
        all_ics = np.array(all_ics)
        all_z = np.array(all_z)
        all_snap = np.array(all_snap)
        all_galaxyindex = np.array(all_galaxyindex)

        p99 = np.percentile(all_ratios, 99)
        global_max = np.max(all_ratios)
        # Use 99th percentile with 10% padding, rounded up
        ratio_max = np.ceil(p99 * 1.1)
        ratio_max = max(ratio_max, 1.0)  # Ensure at least 1.0
        n_outliers = np.sum(all_ratios > ratio_max)
        print(f'  99th percentile: {p99:.2f}, max: {global_max:.2f}, using ratio_max: {ratio_max:.1f}')
        print(f'  Outliers beyond ratio_max: {n_outliers} ({100*n_outliers/len(all_ratios):.1f}%)')

        # Diagnostic: show properties of extreme outliers (top 10 by ratio)
        if n_outliers > 0:
            outlier_mask = all_ratios > ratio_max
            outlier_ratios = all_ratios[outlier_mask]
            outlier_mvir = all_mvir[outlier_mask]
            outlier_stellar = all_stellar[outlier_mask]
            outlier_ics = all_ics[outlier_mask]
            outlier_z = all_z[outlier_mask]
            outlier_snap = all_snap[outlier_mask]
            outlier_galaxyindex = all_galaxyindex[outlier_mask]

            # Sort by ratio (descending) and show top 10
            sort_idx = np.argsort(outlier_ratios)[::-1]
            n_show = min(10, len(sort_idx))

            print(f'\n  Top {n_show} outliers (sorted by M_ICS/M_BCG ratio):')
            print(f'  {"Ratio":>8} {"log Mvir":>10} {"log M_BCG":>10} {"log M_ICS":>10} {"Snap":>6} {"z":>6} {"GalaxyIndex":>12}')
            print(f'  {"-"*8} {"-"*10} {"-"*10} {"-"*10} {"-"*6} {"-"*6} {"-"*12}')
            for i in range(n_show):
                idx = sort_idx[i]
                print(f'  {outlier_ratios[idx]:8.2f} {np.log10(outlier_mvir[idx]):10.2f} '
                      f'{np.log10(outlier_stellar[idx]):10.2f} {np.log10(outlier_ics[idx]):10.2f} '
                      f'{outlier_snap[idx]:6d} {outlier_z[idx]:6.2f} {outlier_galaxyindex[idx]:12d}')

            # Summary statistics for outliers
            print(f'\n  Outlier summary:')
            print(f'    log Mvir range: {np.log10(outlier_mvir.min()):.2f} - {np.log10(outlier_mvir.max()):.2f}')
            print(f'    log M_BCG range: {np.log10(outlier_stellar.min()):.2f} - {np.log10(outlier_stellar.max()):.2f}')
            print(f'    log M_ICS range: {np.log10(outlier_ics.min()):.2f} - {np.log10(outlier_ics.max()):.2f}')
            print(f'    Redshift range: {outlier_z.min():.2f} - {outlier_z.max():.2f}')

            # Print traceable info for merger tree analysis
            print(f'\n  For merger tree tracing, use these (Snap, GalaxyIndex) pairs:')
            for i in range(n_show):
                idx = sort_idx[i]
                print(f'    Snap {outlier_snap[idx]:d}, GalaxyIndex {outlier_galaxyindex[idx]:d} (ratio={outlier_ratios[idx]:.2f})')
    else:
        ratio_max = 10.0  # Fallback
    binwidth = 0.2
    NB = int((ratio_max - ratio_min) / binwidth)

    # Halo mass bin edges (in log10 Msun)
    halo_bin_edges = [10.5, 12, 13.5, 17]
    subset_colors = ['red', 'green', 'dodgerblue']
    legend_labels = [r'$10^{10.5} < M_{\mathrm{vir}} < 10^{12}$',
                     r'$10^{12} < M_{\mathrm{vir}} < 10^{13.5}$',
                     r'$M_{\mathrm{vir}} > 10^{13.5}$']

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    # Track statistics
    total_galaxies = 0
    snap_stats = []

    for idx, snap in enumerate(snaps):
        if idx >= len(axes):
            break

        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data from all files (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS = data['Mvir'], data['IntraClusterStars']
            StellarMass, Type = data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select central galaxies with valid data (minimum threshold for overall)
        base_mask = ((Type == 0) & (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) &
                     (Mvir >= MVIR_THRESHOLD_MASSFUNCTION) & (n_satellites >= MIN_SATELLITES))

        # Apply pathological BCG filter (same as first pass)
        if FILTER_PATHOLOGICAL_BCGS:
            bcg_mvir_ratio = np.where(Mvir > 0, StellarMass / Mvir, 1.0)
            valid_mask = base_mask & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
        else:
            valid_mask = base_mask

        w = np.where(valid_mask)[0]
        total_galaxies += len(w)
        snap_stats.append((snap, z, len(w)))

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(ratio_min, ratio_max)
            ax.set_ylim(1e-8, 1e-1)
            ax.set_yscale('log')
            continue

        Mvir_sel = Mvir[w]
        ICS_sel = ICS[w]
        StellarMass_sel = StellarMass[w]
        log_Mvir = np.log10(Mvir_sel)

        # Calculate ratio
        ics_bcg_ratio = ICS_sel / StellarMass_sel

        # Plot distribution for each halo mass bin
        for bin_idx in range(len(halo_bin_edges) - 1):
            lo = halo_bin_edges[bin_idx]
            hi = halo_bin_edges[bin_idx + 1]
            mask = (log_Mvir >= lo) & (log_Mvir < hi)
            ratio_in_bin = ics_bcg_ratio[mask]

            if len(ratio_in_bin) > 0:
                counts, binedges = np.histogram(ratio_in_bin, range=(ratio_min, ratio_max), bins=NB)
                xaxeshisto = binedges[:-1] + 0.5 * binwidth
                phi = counts / volume / binwidth
                ax.plot(xaxeshisto, phi, color=subset_colors[bin_idx], alpha=0.7)
                ax.fill_between(xaxeshisto, 0, phi, color=subset_colors[bin_idx], alpha=0.1)

        # Plot overall distribution
        counts, binedges = np.histogram(ics_bcg_ratio, range=(ratio_min, ratio_max), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth
        phi = counts / volume / binwidth
        ax.plot(xaxeshisto, phi, color='black')

        # Formatting
        ax.set_yscale('log')
        ax.set_xlim(ratio_min, ratio_max)
        ax.set_ylim(1e-8, 1e-1)

        # Add redshift label (upper right)
        ax.text(0.95, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    # Hide any unused axes
    for idx in range(len(snaps), len(axes)):
        axes[idx].set_visible(False)

    # Add x-axis labels at bottom of each column
    axes[6].set_xlabel(r'$M_{\mathrm{ICS}} / M_{\mathrm{BCG}}$', fontsize=14)
    axes[7].set_xlabel(r'$M_{\mathrm{ICS}} / M_{\mathrm{BCG}}$', fontsize=14)

    # Add y-axis labels for each row (left column only)
    ylabel = r'$\log_{10}\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$'
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[2].set_ylabel(ylabel, fontsize=12)
    axes[4].set_ylabel(ylabel, fontsize=12)
    axes[6].set_ylabel(ylabel, fontsize=12)

    # Create legend at bottom of figure
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in subset_colors]
    custom_lines.append(Line2D([0], [0], color='black', lw=2))
    legend_labels_with_overall = legend_labels + [f'Overall ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_MASSFUNCTION):.1f}}}$ M$_\\odot$)']
    fig.legend(custom_lines, legend_labels_with_overall,
               loc='lower center', ncol=4, fontsize=10, frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05)

    outputFile = os.path.join(output_dir, 'ICS_BCG_ratio_distribution' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps)} (z = {redshifts[snaps[0]]:.2f} to {redshifts[snaps[-1]]:.2f})')
    print(f'  Total centrals plotted: {total_galaxies}')
    for snap, z, n in snap_stats:
        print(f'    z = {z:.2f}: {n} galaxies')
    print(f'  Saved: {outputFile}')


def plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir, max_redshift=2.5, cache=None):
    """
    Plot ICS fraction (M_ICS / (f_b * M_vir)) vs redshift as a scatter plot.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    max_redshift : float
        Maximum redshift to include (default 1.5)
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Fraction vs Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Filter snapshots to those with z <= max_redshift
    snaps_to_plot = [s for s in available_snapshots
                     if s < len(redshifts) and redshifts[s] <= max_redshift]
    snaps_to_plot.sort(reverse=True)  # Start from highest redshift

    if not snaps_to_plot:
        print('  No snapshots available in redshift range.')
        return

    # Collect data from all snapshots
    all_redshifts = []
    group_redshifts = []
    clusters_ics_fractions = []
    clusters_mvir = []
    groups_ics_fractions = []
    groups_mvir = []

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS, StellarMass, Type = data['Mvir'], data['IntraClusterStars'], data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with valid data (applying thresholds) for CLUSTERS
        clusters = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]

        if len(clusters) == 0:
            continue

        # Calculate ICS fraction
        ics_fraction = ICS[clusters] / (BARYON_FRACTION * Mvir[clusters])

        # Store data
        all_redshifts.extend([z] * len(clusters))
        clusters_ics_fractions.extend(ics_fraction)
        clusters_mvir.extend(Mvir[clusters])

    if not all_redshifts:
        print('  No data to plot.')
        return

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS, StellarMass, Type = data['Mvir'], data['IntraClusterStars'], data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with valid data (applying thresholds) for GROUPS
        groups = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT_GROUPS) & (Mvir < MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]

        if len(groups) == 0:
            continue

        # Calculate ICS fraction
        ics_fraction = ICS[groups] / (BARYON_FRACTION * Mvir[groups])

        # Store data
        group_redshifts.extend([z] * len(groups))
        groups_ics_fractions.extend(ics_fraction)
        groups_mvir.extend(Mvir[groups])

    if not all_redshifts:
        print('  No data to plot.')
        return

    all_redshifts = np.array(all_redshifts)
    group_redshifts = np.array(group_redshifts)
    clusters_ics_fractions = np.array(clusters_ics_fractions)
    clusters_mvir = np.array(clusters_mvir)
    groups_ics_fractions = np.array(groups_ics_fractions)
    groups_mvir = np.array(groups_mvir)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate max at each redshift for the max line
    unique_redshifts = np.sort(np.unique(all_redshifts))
    max_vals_clusters = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            max_vals_clusters.append(np.max(clusters_ics_fractions[mask]))

    max_vals_clusters = np.array(max_vals_clusters)

    # Plot max line
    # ax.plot(unique_redshifts, max_vals_clusters, 'r--', lw=1.5, label=r'Max $f_{\mathrm{ICS}}$')

    # Calculate median and 1-sigma (standard deviation) at each redshift
    median_vals_clusters = []
    sigma_lo_clusters = []
    sigma_hi_clusters = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            vals = clusters_ics_fractions[mask]
            
            med = np.median(vals)
            median_vals_clusters.append(med)
            
            # 1-sigma equivalent percentiles of the raw data
            sigma_lo_clusters.append(np.percentile(vals, 16))
            sigma_hi_clusters.append(np.percentile(vals, 84))

    median_vals_clusters = np.array(median_vals_clusters)
    sigma_lo_clusters = np.array(sigma_lo_clusters)
    sigma_hi_clusters = np.array(sigma_hi_clusters)

    # Plot median line with 1-sigma error band
    ax.plot(unique_redshifts, median_vals_clusters, '-', color='dodgerblue', lw=1.5, label=r'Clusters: Median $f_{\mathrm{ICS}}$')
    ax.fill_between(unique_redshifts, sigma_lo_clusters, sigma_hi_clusters, color='dodgerblue', alpha=0.15)

    max_vals_groups = []
    unique_redshifts_group = np.sort(np.unique(group_redshifts))

    for z in unique_redshifts_group:
        mask = group_redshifts == z
        if np.any(mask):
            max_vals_groups.append(np.max(groups_ics_fractions[mask]))

    max_vals_groups = np.array(max_vals_groups)

    # Plot max line
    # ax.plot(unique_redshifts_group, max_vals_groups, 'b--', lw=1.5, label=r'Max $f_{\mathrm{ICS}}$ (Groups)')

    # Calculate median and 1-sigma (standard deviation) at each redshift
    median_vals_groups = []
    sigma_lo_groups = []
    sigma_hi_groups = []

    for z in unique_redshifts:
        mask = group_redshifts == z
        if np.any(mask):
            vals = groups_ics_fractions[mask]
            
            med = np.median(vals)
            median_vals_groups.append(med)
            
            # 1-sigma equivalent percentiles of the raw data
            sigma_lo_groups.append(np.percentile(vals, 16))
            sigma_hi_groups.append(np.percentile(vals, 84))

    median_vals_groups = np.array(median_vals_groups)
    sigma_lo_groups = np.array(sigma_lo_groups)
    sigma_hi_groups = np.array(sigma_hi_groups)

    # Plot median line with 1-sigma error band
    ax.plot(unique_redshifts_group, median_vals_groups, '--', color='green', lw=1.5, label=r'Groups: Median $f_{\mathrm{ICS}}$')
    ax.fill_between(unique_redshifts_group, sigma_lo_groups, sigma_hi_groups, color='green', alpha=0.15)


    # Plot stratified sample of points at each redshift to show full spread
    # np.random.seed(42)
    # n_bins = 2  # Number of f_ICS bins for stratified sampling
    # n_per_bin = 1  # Points to sample from each bin
    # for z in unique_redshifts:
    #     mask = all_redshifts == z
    #     indices = np.where(mask)[0]
    #     fracs = all_ics_fractions[indices]

    #     # Create bins across the f_ICS range for this snapshot
    #     frac_min, frac_max = fracs.min(), fracs.max()
    #     if frac_max > frac_min:
    #         bin_edges = np.linspace(frac_min, frac_max, n_bins + 1)
    #         sampled_indices = []
    #         for i in range(n_bins):
    #             bin_mask = (fracs >= bin_edges[i]) & (fracs < bin_edges[i + 1])
    #             if i == n_bins - 1:  # Include max value in last bin
    #                 bin_mask = (fracs >= bin_edges[i]) & (fracs <= bin_edges[i + 1])
    #             bin_indices = indices[bin_mask]
    #             if len(bin_indices) > 0:
    #                 n_sample = min(n_per_bin, len(bin_indices))
    #                 sampled_indices.extend(np.random.choice(bin_indices, n_sample, replace=False))
    #         indices = np.array(sampled_indices)

    #     ax.scatter(all_redshifts[indices], all_ics_fractions[indices], s=40, alpha=0.6,
    #                facecolor='dodgerdodgerblue', edgecolor='black', rasterized=True, marker='H', zorder=5)

    # # Add legend entry for scatter points
    # ax.scatter([], [], s=40, facecolor='dodgerdodgerblue', edgecolor='black', marker='H',
    #            label=f'SAGE26 ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_REDSHIFT):.1f}}}$ M$_\\odot$)')

    # Add observational data and simulation comparisons
    plot_ics_observations(ax)

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14)
    ax.set_xlim(0, max_redshift)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_fraction_vs_redshift' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps_to_plot)} (z = 0 to {max_redshift})')
    print(f'  Total data points: (Clusters: {len(clusters_ics_fractions)}, Groups: {len(groups_ics_fractions)})')
    print(f'  ICS fraction: median = {np.median(clusters_ics_fractions):.3f}, '
          f'mean = {np.mean(clusters_ics_fractions):.3f}, std = {np.std(clusters_ics_fractions):.3f}')
    print(f'  Saved: {outputFile}')


def plot_ics_stellar_fraction_vs_redshift(file_list, sim_params, output_dir, max_redshift=2.5, cache=None):
    """
    Plot ICS fraction (M_ICS / total stellar mass within Rvir) vs redshift.

    The denominator is the total stellar mass within the virial radius,
    i.e. f_ICS = M_ICS / (M_*,BCG + sum(M_*,sat) + M_ICS).

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    max_redshift : float
        Maximum redshift to include (default 2.5)
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Stellar Fraction vs Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Filter snapshots to those with z <= max_redshift
    snaps_to_plot = [s for s in available_snapshots
                     if s < len(redshifts) and redshifts[s] <= max_redshift]
    snaps_to_plot.sort(reverse=True)  # Start from highest redshift

    if not snaps_to_plot:
        print('  No snapshots available in redshift range.')
        return

    # Collect data from all snapshots
    all_redshifts = []
    group_redshifts = []
    clusters_ics_fractions = []
    clusters_mvir = []
    groups_ics_fractions = []
    groups_mvir = []

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type',
                                        'GalaxyIndex', 'CentralGalaxyIndex'])
            Mvir, ICS, StellarMass, Type = data['Mvir'], data['IntraClusterStars'], data['StellarMass'], data['Type']
            GalaxyIndex, CentralGalaxyIndex = data['GalaxyIndex'], data['CentralGalaxyIndex']
            satellite_mass = cache.get_satellite_mass(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            # Compute satellite mass summed to centrals
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            satellite_mask_tmp = Type != 0
            satellite_central_gids = CentralGalaxyIndex[satellite_mask_tmp]
            satellite_masses = StellarMass[satellite_mask_tmp]
            insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
            insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
            valid_match = sorted_gids[insert_pos] == satellite_central_gids
            central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
            valid_satellites = central_indices >= 0
            satellite_mass = np.zeros(len(StellarMass))
            np.add.at(satellite_mass, central_indices[valid_satellites], satellite_masses[valid_satellites])

        # Count satellites per central
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_mask = Type != 0
        sat_central_gids = CentralGalaxyIndex[sat_mask]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        valid = sorted_gids[ins_pos] == sat_central_gids
        c_idx = np.where(valid, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(StellarMass), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Total stellar mass within Rvir (BCG + satellites + ICS)
        total_stellar = StellarMass + satellite_mass + ICS

        # BCG/Mvir ratio for pathological BCG filter
        bcg_mvir_ratio = np.where(Mvir > 0, StellarMass / Mvir, 0)

        # Select centrals with valid data (applying thresholds) for CLUSTERS
        base_mask = ((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0) &
                     (n_satellites >= MIN_SATELLITES))
        if FILTER_PATHOLOGICAL_BCGS:
            base_mask = base_mask & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
        clusters = np.where(base_mask)[0]

        if len(clusters) == 0:
            continue

        # Calculate ICS fraction: M_ICS / total stellar mass within Rvir
        ics_fraction = ICS[clusters] / total_stellar[clusters]

        # Store data
        all_redshifts.extend([z] * len(clusters))
        clusters_ics_fractions.extend(ics_fraction)
        clusters_mvir.extend(Mvir[clusters])

    if not all_redshifts:
        print('  No data to plot.')
        return

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type',
                                        'GalaxyIndex', 'CentralGalaxyIndex'])
            Mvir, ICS, StellarMass, Type = data['Mvir'], data['IntraClusterStars'], data['StellarMass'], data['Type']
            GalaxyIndex, CentralGalaxyIndex = data['GalaxyIndex'], data['CentralGalaxyIndex']
            satellite_mass = cache.get_satellite_mass(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            satellite_mask_tmp = Type != 0
            satellite_central_gids = CentralGalaxyIndex[satellite_mask_tmp]
            satellite_masses = StellarMass[satellite_mask_tmp]
            insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
            insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
            valid_match = sorted_gids[insert_pos] == satellite_central_gids
            central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
            valid_satellites = central_indices >= 0
            satellite_mass = np.zeros(len(StellarMass))
            np.add.at(satellite_mass, central_indices[valid_satellites], satellite_masses[valid_satellites])

        # Count satellites per central
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_mask = Type != 0
        sat_central_gids = CentralGalaxyIndex[sat_mask]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        valid = sorted_gids[ins_pos] == sat_central_gids
        c_idx = np.where(valid, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(StellarMass), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Total stellar mass within Rvir (BCG + satellites + ICS)
        total_stellar = StellarMass + satellite_mass + ICS

        # BCG/Mvir ratio for pathological BCG filter
        bcg_mvir_ratio = np.where(Mvir > 0, StellarMass / Mvir, 0)

        # Select centrals with valid data (applying thresholds) for GROUPS
        base_mask = ((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT_GROUPS) & (Mvir < MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0) &
                     (n_satellites >= MIN_SATELLITES))
        if FILTER_PATHOLOGICAL_BCGS:
            base_mask = base_mask & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
        groups = np.where(base_mask)[0]

        if len(groups) == 0:
            continue

        # Calculate ICS fraction: M_ICS / total stellar mass within Rvir
        ics_fraction = ICS[groups] / total_stellar[groups]

        # Store data
        group_redshifts.extend([z] * len(groups))
        groups_ics_fractions.extend(ics_fraction)
        groups_mvir.extend(Mvir[groups])

    if not all_redshifts:
        print('  No data to plot.')
        return

    all_redshifts = np.array(all_redshifts)
    group_redshifts = np.array(group_redshifts)
    clusters_ics_fractions = np.array(clusters_ics_fractions)
    clusters_mvir = np.array(clusters_mvir)
    groups_ics_fractions = np.array(groups_ics_fractions)
    groups_mvir = np.array(groups_mvir)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate max at each redshift for the max line
    unique_redshifts = np.sort(np.unique(all_redshifts))
    max_vals_clusters = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            max_vals_clusters.append(np.max(clusters_ics_fractions[mask]))

    max_vals_clusters = np.array(max_vals_clusters)

    # Plot max line
    # ax.plot(unique_redshifts, max_vals_clusters, 'r--', lw=1.5, label=r'Max $f_{\mathrm{ICS}}$')

    # Calculate median and 1-sigma (standard deviation) at each redshift
    median_vals_clusters = []
    sigma_lo_clusters = []
    sigma_hi_clusters = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            vals = clusters_ics_fractions[mask]
            
            med = np.median(vals)
            median_vals_clusters.append(med)
            
            # 1-sigma equivalent percentiles of the raw data
            sigma_lo_clusters.append(np.percentile(vals, 16))
            sigma_hi_clusters.append(np.percentile(vals, 84))

    median_vals_clusters = np.array(median_vals_clusters)
    sigma_lo_clusters = np.array(sigma_lo_clusters)
    sigma_hi_clusters = np.array(sigma_hi_clusters)

    # Plot median line with 1-sigma error band
    ax.plot(unique_redshifts, median_vals_clusters, '-', color='dodgerblue', lw=1.5, label=r'Clusters: Median $f_{\mathrm{ICS}}$')
    ax.fill_between(unique_redshifts, sigma_lo_clusters, sigma_hi_clusters, color='dodgerblue', alpha=0.15)

    max_vals_groups = []
    unique_redshifts_group = np.sort(np.unique(group_redshifts))

    for z in unique_redshifts_group:
        mask = group_redshifts == z
        if np.any(mask):
            max_vals_groups.append(np.max(groups_ics_fractions[mask]))

    max_vals_groups = np.array(max_vals_groups)

    # Plot max line
    # ax.plot(unique_redshifts_group, max_vals_groups, 'b--', lw=1.5, label=r'Max $f_{\mathrm{ICS}}$ (Groups)')

    # Calculate median and 1-sigma (standard deviation) at each redshift
    median_vals_groups = []
    sigma_lo_groups = []
    sigma_hi_groups = []

    for z in unique_redshifts:
        mask = group_redshifts == z
        if np.any(mask):
            vals = groups_ics_fractions[mask]
            
            med = np.median(vals)
            median_vals_groups.append(med)
            
            # 1-sigma equivalent percentiles of the raw data
            sigma_lo_groups.append(np.percentile(vals, 16))
            sigma_hi_groups.append(np.percentile(vals, 84))

    median_vals_groups = np.array(median_vals_groups)
    sigma_lo_groups = np.array(sigma_lo_groups)
    sigma_hi_groups = np.array(sigma_hi_groups)

    # Plot median line with 1-sigma error band
    ax.plot(unique_redshifts_group, median_vals_groups, 'g--', lw=1.5, label=r'Groups: Median $f_{\mathrm{ICS}}$')
    ax.fill_between(unique_redshifts_group, sigma_lo_groups, sigma_hi_groups, color='green', alpha=0.15)

    # Add observational data and simulation comparisons
    plot_ics_observations(ax)

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (M_{*,\mathrm{BCG}} + M_{*,\mathrm{sat}} + M_{\mathrm{ICS}})$', fontsize=14)
    ax.set_xlim(0, max_redshift)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_stellar_fraction_vs_redshift' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps_to_plot)} (z = 0 to {max_redshift})')
    print(f'  Total data points: (Clusters: {len(clusters_ics_fractions)}, Groups: {len(groups_ics_fractions)})')
    print(f'  ICS stellar fraction: median = {np.median(clusters_ics_fractions):.3f}, '
          f'mean = {np.mean(clusters_ics_fractions):.3f}, std = {np.std(clusters_ics_fractions):.3f}')
    print(f'  Saved: {outputFile}')


def plot_ics_bcg_fraction_vs_redshift(file_list, sim_params, output_dir, max_redshift=2.5, cache=None):
    """
    Plot ICS+BCG fraction ((M_ICS + M_stellar) / (f_b * M_vir)) vs redshift as a scatter plot.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    max_redshift : float
        Maximum redshift to include (default 1.5)
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS+BCG Fraction vs Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Filter snapshots to those with z <= max_redshift
    snaps_to_plot = [s for s in available_snapshots
                     if s < len(redshifts) and redshifts[s] <= max_redshift]
    snaps_to_plot.sort(reverse=True)  # Start from highest redshift

    if not snaps_to_plot:
        print('  No snapshots available in redshift range.')
        return

    # Collect data from all snapshots
    all_redshifts = []
    all_fractions = []
    all_mvir = []
    group_redshifts = []
    groups_ics_fractions = []


    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS = data['Mvir'], data['IntraClusterStars']
            StellarMass, Type = data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with valid data (applying thresholds)
        w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) &
                     (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]
        
        w_groups = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT_GROUPS) & (Mvir < MVIR_THRESHOLD_REDSHIFT) &
                     (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]

        if len(w) == 0:
            continue

        # Calculate ICS+BCG fraction
        ics_bcg_fraction = (ICS[w] + StellarMass[w]) / (BARYON_FRACTION * Mvir[w])
        ics_bcg_fraction_groups = (ICS[w_groups] + StellarMass[w_groups]) / (BARYON_FRACTION * Mvir[w_groups])

        # Store data
        all_redshifts.extend([z] * len(w))
        all_fractions.extend(ics_bcg_fraction)
        all_mvir.extend(Mvir[w])
        group_redshifts.extend([z] * len(w_groups))
        groups_ics_fractions.extend(ics_bcg_fraction_groups)

    if not all_redshifts:
        print('  No data to plot.')
        return

    all_redshifts = np.array(all_redshifts)
    all_fractions = np.array(all_fractions)
    all_mvir = np.array(all_mvir)
    group_redshifts = np.array(group_redshifts)
    groups_ics_fractions = np.array(groups_ics_fractions)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate max at each redshift for the max line
    unique_redshifts = np.sort(np.unique(all_redshifts))
    max_vals = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            max_vals.append(np.max(all_fractions[mask]))

    max_vals = np.array(max_vals)

    # # Plot max line
    # ax.plot(unique_redshifts, max_vals, 'r--', lw=1.5, label=r'Max $f_{\mathrm{ICS+BCG}}$')

    # # Plot stratified sample of points at each redshift to show full spread
    # np.random.seed(42)
    # n_bins = 2  # Number of fraction bins for stratified sampling
    # n_per_bin = 1  # Points to sample from each bin
    # for z in unique_redshifts:
    #     mask = all_redshifts == z
    #     indices = np.where(mask)[0]
    #     fracs = all_fractions[indices]

    #     # Create bins across the fraction range for this snapshot
    #     frac_min, frac_max = fracs.min(), fracs.max()
    #     if frac_max > frac_min:
    #         bin_edges = np.linspace(frac_min, frac_max, n_bins + 1)
    #         sampled_indices = []
    #         for i in range(n_bins):
    #             bin_mask = (fracs >= bin_edges[i]) & (fracs < bin_edges[i + 1])
    #             if i == n_bins - 1:  # Include max value in last bin
    #                 bin_mask = (fracs >= bin_edges[i]) & (fracs <= bin_edges[i + 1])
    #             bin_indices = indices[bin_mask]
    #             if len(bin_indices) > 0:
    #                 n_sample = min(n_per_bin, len(bin_indices))
    #                 sampled_indices.extend(np.random.choice(bin_indices, n_sample, replace=False))
    #         indices = np.array(sampled_indices)

    #     ax.scatter(all_redshifts[indices], all_fractions[indices], s=40, alpha=0.6,
    #                facecolor='dodgerdodgerblue', edgecolor='black', rasterized=True, marker='H', zorder=5)

    # # Add legend entry for scatter points
    # ax.scatter([], [], s=40, facecolor='dodgerdodgerblue', edgecolor='black', marker='H',
    #            label=f'SAGE26 ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_REDSHIFT):.1f}}}$ M$_\\odot$)')

    # Plot median line of clusters with 1-sigma error band
    median_vals = []
    sigma_lo = []
    sigma_hi = []

    for z in unique_redshifts:
        mask = all_redshifts == z
        if np.any(mask):
            vals = all_fractions[mask]
            median_vals.append(np.median(vals))
            sigma_lo.append(np.percentile(vals, 16))
            sigma_hi.append(np.percentile(vals, 84))

    median_vals = np.array(median_vals)
    sigma_lo = np.array(sigma_lo)
    sigma_hi = np.array(sigma_hi)

    ax.plot(unique_redshifts, median_vals, '-', color='dodgerblue', lw=1.5, label=r'Clusters: Median $f_{\mathrm{ICS+BCG}}$')
    ax.fill_between(unique_redshifts, sigma_lo, sigma_hi, color='dodgerblue', alpha=0.15)

    # Plot median line of groups with 1-sigma error band
    group_median_vals = []
    group_sigma_lo = []
    group_sigma_hi = []

    for z in unique_redshifts:
        mask = group_redshifts == z
        if np.any(mask):
            vals = groups_ics_fractions[mask]
            group_median_vals.append(np.median(vals))
            group_sigma_lo.append(np.percentile(vals, 16))
            group_sigma_hi.append(np.percentile(vals, 84))

    group_median_vals = np.array(group_median_vals)
    group_sigma_lo = np.array(group_sigma_lo)
    group_sigma_hi = np.array(group_sigma_hi)

    ax.plot(unique_redshifts, group_median_vals, 'g--', lw=1.5, label=r'Groups: Median $f_{\mathrm{ICS+BCG}}$')
    ax.fill_between(unique_redshifts, group_sigma_lo, group_sigma_hi, color='green', alpha=0.15)

    # Add observational data and simulation comparisons
    plot_ics_observations(ax)

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS+BCG}} = (M_{\mathrm{ICS}} + M_{\star}) / (f_b \times M_{\mathrm{vir}})$', fontsize=14)
    ax.set_xlim(0, max_redshift)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_BCG_fraction_vs_redshift' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps_to_plot)} (z = 0 to {max_redshift})')
    print(f'  Total data points: {len(all_fractions)}')
    print(f'  ICS+BCG fraction: median = {np.median(all_fractions):.3f}, '
          f'mean = {np.mean(all_fractions):.3f}, std = {np.std(all_fractions):.3f}')
    print(f'  Saved: {outputFile}')


def plot_ics_bcg_ratio_vs_redshift(file_list, sim_params, output_dir, max_redshift=2.0, cache=None):
    """
    Plot M_ICS / M_BCG ratio vs redshift as a scatter plot.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    max_redshift : float
        Maximum redshift to include (default 2.0)
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS/BCG Ratio vs Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Filter snapshots to those with z <= max_redshift
    snaps_to_plot = [s for s in available_snapshots
                     if s < len(redshifts) and redshifts[s] <= max_redshift]
    snaps_to_plot.sort(reverse=True)  # Start from highest redshift

    if not snaps_to_plot:
        print('  No snapshots available in redshift range.')
        return

    # Collect data from all snapshots
    all_redshifts = []
    all_ratios = []
    all_ics_mass = []
    all_bcg_mass = []
    all_mvir = []

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data (using cache if available)
        if cache is not None:
            data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'])
            Mvir, ICS = data['Mvir'], data['IntraClusterStars']
            StellarMass, Type = data['StellarMass'], data['Type']
            n_satellites = cache.get_satellite_count(Snapshot)
        else:
            Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
            ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
            StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
            Type = read_hdf(file_list, Snapshot, 'Type')
            GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
            CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
            sorted_idx = np.argsort(GalaxyIndex)
            sorted_gids = GalaxyIndex[sorted_idx]
            sat_central_gids = CentralGalaxyIndex[Type != 0]
            ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
            ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
            c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
            n_satellites = np.zeros(len(Type), dtype=int)
            np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with valid data (applying thresholds)
        # Require both ICS > 0 and StellarMass > threshold for meaningful ratio
        base_mask = ((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) &
                     (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))

        # Apply pathological BCG filter
        if FILTER_PATHOLOGICAL_BCGS:
            bcg_mvir_ratio = np.where(Mvir > 0, StellarMass / Mvir, 1.0)
            valid_mask = base_mask & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
        else:
            valid_mask = base_mask

        w = np.where(valid_mask)[0]

        if len(w) == 0:
            continue

        # Calculate M_ICS / M_BCG ratio
        ics_bcg_ratio = ICS[w] / StellarMass[w]

        # Store data
        all_redshifts.extend([z] * len(w))
        all_ratios.extend(ics_bcg_ratio)
        all_ics_mass.extend(ICS[w])
        all_bcg_mass.extend(StellarMass[w])
        all_mvir.extend(Mvir[w])

    if not all_redshifts:
        print('  No data to plot.')
        return

    all_redshifts = np.array(all_redshifts)
    all_ratios = np.array(all_ratios)
    all_ics_mass = np.array(all_ics_mass)
    all_bcg_mass = np.array(all_bcg_mass)
    all_mvir = np.array(all_mvir)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot representative points at each redshift
    unique_redshifts = np.unique(all_redshifts)
    for z in unique_redshifts:
        mask = all_redshifts == z
        indices = np.where(mask)[0]
        # Plot up to 10 points per redshift
        n_points = min(10, len(indices))
        sample_idx = np.random.choice(indices, n_points, replace=False) if len(indices) > n_points else indices
        ax.scatter(all_redshifts[sample_idx], all_ratios[sample_idx], s=75, alpha=1.0,
                   facecolor='dodgerblue', edgecolor='black', rasterized=True, marker='H')

    # Add legend entry
    ax.scatter([], [], s=75, alpha=1.0, facecolor='dodgerblue', edgecolor='black', marker='H',
               label=f'SAGE26 ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_REDSHIFT):.1f}}}$ M$_\\odot$)')

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'$M_{\mathrm{ICS}} / M_{\mathrm{BCG}}$', fontsize=14)
    ax.set_xlim(0, max_redshift)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_BCG_ratio_vs_redshift' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshots: {len(snaps_to_plot)} (z = 0 to {max_redshift})')
    print(f'  Total data points: {len(all_ratios)}')
    print(f'  M_ICS/M_BCG ratio statistics:')
    print(f'    Min: {np.min(all_ratios):.3f}, Max: {np.max(all_ratios):.3f}')
    print(f'    Mean: {np.mean(all_ratios):.3f}, Median: {np.median(all_ratios):.3f}, Std: {np.std(all_ratios):.3f}')
    print(f'    Percentiles: 10th={np.percentile(all_ratios, 10):.3f}, '
          f'25th={np.percentile(all_ratios, 25):.3f}, 75th={np.percentile(all_ratios, 75):.3f}, '
          f'90th={np.percentile(all_ratios, 90):.3f}, 99th={np.percentile(all_ratios, 99):.3f}')
    print(f'  Distribution:')
    print(f'    Ratio < 0.5: {np.sum(all_ratios < 0.5)} ({100*np.sum(all_ratios < 0.5)/len(all_ratios):.1f}%)')
    print(f'    Ratio 0.5-1: {np.sum((all_ratios >= 0.5) & (all_ratios < 1))}'
          f' ({100*np.sum((all_ratios >= 0.5) & (all_ratios < 1))/len(all_ratios):.1f}%)')
    print(f'    Ratio 1-2:   {np.sum((all_ratios >= 1) & (all_ratios < 2))}'
          f' ({100*np.sum((all_ratios >= 1) & (all_ratios < 2))/len(all_ratios):.1f}%)')
    print(f'    Ratio 2-5:   {np.sum((all_ratios >= 2) & (all_ratios < 5))}'
          f' ({100*np.sum((all_ratios >= 2) & (all_ratios < 5))/len(all_ratios):.1f}%)')
    print(f'    Ratio > 5:   {np.sum(all_ratios >= 5)} ({100*np.sum(all_ratios >= 5)/len(all_ratios):.1f}%)')

    # Diagnose extreme outliers
    extreme_thresholds = [100, 50, 20]
    for thresh in extreme_thresholds:
        extreme_mask = all_ratios > thresh
        n_extreme = np.sum(extreme_mask)
        if n_extreme > 0:
            print(f'  Extreme outliers (ratio > {thresh}): {n_extreme} ({100*n_extreme/len(all_ratios):.2f}%)')
            # Sort by ratio and show worst cases
            extreme_idx = np.where(extreme_mask)[0]
            sort_order = np.argsort(all_ratios[extreme_idx])[::-1]  # Descending
            extreme_idx_sorted = extreme_idx[sort_order]
            print(f'    Top 5 most extreme:')
            for i, idx in enumerate(extreme_idx_sorted[:5]):
                print(f'      {i+1}. z={all_redshifts[idx]:.2f}, ratio={all_ratios[idx]:.1f}, '
                      f'M_ICS={all_ics_mass[idx]:.2e}, M_BCG={all_bcg_mass[idx]:.2e}, '
                      f'log(Mvir)={np.log10(all_mvir[idx]):.2f}')
            break  # Only show details for the highest threshold with outliers

    # Summary of mass ranges for high vs low ratio systems
    print(f'  Mass comparison (low vs high ratio):')
    low_ratio_mask = all_ratios < 1
    high_ratio_mask = all_ratios > 5
    if np.any(low_ratio_mask) and np.any(high_ratio_mask):
        print(f'    Low ratio (< 1):  median M_BCG={np.median(all_bcg_mass[low_ratio_mask]):.2e}, '
              f'median M_ICS={np.median(all_ics_mass[low_ratio_mask]):.2e}, '
              f'median log(Mvir)={np.median(np.log10(all_mvir[low_ratio_mask])):.2f}')
        print(f'    High ratio (> 5): median M_BCG={np.median(all_bcg_mass[high_ratio_mask]):.2e}, '
              f'median M_ICS={np.median(all_ics_mass[high_ratio_mask]):.2e}, '
              f'median log(Mvir)={np.median(np.log10(all_mvir[high_ratio_mask])):.2f}')

    print(f'  Saved: {outputFile}')


def plot_ics_bcg_ratio_vs_halomass(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot M_ICS / M_BCG ratio vs halo mass at z=0.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    snap : int, optional
        Snapshot to use. Defaults to latest available (z~0).
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS/BCG Ratio vs Halo Mass Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Use latest snapshot if not specified
    if snap is None:
        snap = sim_params['last_snapshot']

    if snap not in available_snapshots:
        print(f'  Snapshot {snap} not available.')
        return

    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # Read data (using cache if available)
    fields = ['Mvir', 'StellarMass', 'IntraClusterStars', 'Type', 'SAGETreeIndex']
    if cache is not None:
        data = cache.get(Snapshot, fields)
        Mvir = data['Mvir']
        StellarMass = data['StellarMass']
        ICS = data['IntraClusterStars']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        SAGETreeIndex = read_hdf(file_list, Snapshot, 'SAGETreeIndex')
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    # Select centrals with valid data
    base_mask = ((Type == 0) & (Mvir >= MVIR_THRESHOLD_HALOMASS) &
                 (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))

    # Apply pathological BCG filter
    if FILTER_PATHOLOGICAL_BCGS:
        bcg_mvir_ratio = np.where(Mvir > 0, StellarMass / Mvir, 1.0)
        valid_mask = base_mask & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    else:
        valid_mask = base_mask

    w = np.where(valid_mask)[0]

    if len(w) == 0:
        print('  No galaxies meet selection criteria.')
        return

    # Calculate ratio and halo mass
    log_Mvir = np.log10(Mvir[w])
    ics_bcg_ratio = ICS[w] / StellarMass[w]
    tree_indices = SAGETreeIndex[w]

    # Get formation redshifts (cached)
    if cache is not None:
        formation_z = cache.get_formation_redshifts(Snapshot, tree_indices)
    else:
        formation_z = np.zeros(len(tree_indices))

    valid_formation = ~np.isnan(formation_z)
    formation_z_capped = np.clip(formation_z, 0, 1.25)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(log_Mvir[valid_formation], ics_bcg_ratio[valid_formation],
                    c=formation_z_capped[valid_formation], cmap='plasma', vmin=0, vmax=1.25,
                    s=75, alpha=0.8, edgecolor='black', rasterized=True, marker='H')

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$z_{\mathrm{form}}$', fontsize=12)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'$M_{\mathrm{ICS}} / M_{\mathrm{BCG}}$', fontsize=14)

    # Set axis limits
    xlim_min = np.log10(MVIR_THRESHOLD_HALOMASS)
    xlim_max = max(log_Mvir) + 0.2
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(0, None)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_BCG_ratio_vs_halomass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshot: {snap} (z = {z:.2f})')
    print(f'  Centrals plotted: {len(w)}')
    print(f'  Halo mass range: log(Mvir) = {log_Mvir.min():.2f} to {log_Mvir.max():.2f}')
    print(f'  M_ICS/M_BCG ratio: median = {np.median(ics_bcg_ratio):.3f}, '
          f'mean = {np.mean(ics_bcg_ratio):.3f}, std = {np.std(ics_bcg_ratio):.3f}')

    # Pearson correlations
    if len(w) > 1:
        from scipy.stats import pearsonr
        corr_coef, p_value = pearsonr(log_Mvir, ics_bcg_ratio)
        print(f'  Pearson correlation (log Mvir vs ratio): r = {corr_coef:.3f}, p-value = {p_value:.3e}')
        if np.sum(valid_formation) > 1:
            corr_coef_zf, p_value_zf = pearsonr(formation_z[valid_formation], ics_bcg_ratio[valid_formation])
            print(f'  Pearson correlation (z_form vs ratio): r = {corr_coef_zf:.3f}, p-value = {p_value_zf:.3e}')

    print(f'  Formation z: median = {np.nanmedian(formation_z):.2f}, mean = {np.nanmean(formation_z):.2f}, '
          f'min = {np.nanmin(formation_z):.2f}, max = {np.nanmax(formation_z):.2f}')
    n_clipped = np.sum(formation_z > 1.25)
    pct_clipped = 100 * n_clipped / np.sum(valid_formation) if np.sum(valid_formation) > 0 else 0
    print(f'  Formation z > 1.25 (clipped in colorbar): {n_clipped} ({pct_clipped:.1f}%)')
    print(f'  Saved: {outputFile}')


def plot_ics_bcg_fraction_vs_halomass(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot ICS+BCG fraction vs halo mass in two panels:
    Left: scatter plot
    Right: hexbin colored by formation redshift

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    snap : int, optional
        Snapshot to use. Defaults to latest available (z~0).
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS+BCG Fraction vs Halo Mass Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Use latest snapshot if not specified
    if snap is None:
        snap = sim_params['last_snapshot']

    if snap not in available_snapshots:
        print(f'  Snapshot {snap} not available.')
        return

    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # Read data (using cache if available)
    fields = ['Mvir', 'StellarMass', 'IntraClusterStars', 'Type', 'SAGETreeIndex']
    if cache is not None:
        data = cache.get(Snapshot, fields)
        Mvir = data['Mvir']
        StellarMass = data['StellarMass']
        ICS = data['IntraClusterStars']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']
        # Use cached satellite mass calculation
        satellite_mass = cache.get_satellite_mass(Snapshot)
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        SAGETreeIndex = read_hdf(file_list, Snapshot, 'SAGETreeIndex')
        # Fallback: compute satellite mass without cache
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        satellite_mask = Type != 0
        satellite_central_gids = CentralGalaxyIndex[satellite_mask]
        satellite_masses = StellarMass[satellite_mask]
        insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
        insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
        valid_match = sorted_gids[insert_pos] == satellite_central_gids
        central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
        valid_satellites = central_indices >= 0
        satellite_mass = np.zeros(len(Mvir))
        np.add.at(satellite_mass, central_indices[valid_satellites], satellite_masses[valid_satellites])
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, central_indices[valid_satellites], 1)

    # Select centrals with valid data
    w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_HALOMASS) &
                 (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w) == 0:
        print('  No galaxies meet selection criteria.')
        return

    # Calculate masses and fraction
    BCG_mass = StellarMass[w]
    ICS_mass = ICS[w]
    Sat_mass = satellite_mass[w]
    log_Mvir = np.log10(Mvir[w])
    tree_indices = SAGETreeIndex[w]

    M_total = BCG_mass + ICS_mass + Sat_mass
    BCG_ICS_fraction = (BCG_mass + ICS_mass) / M_total

    # Get formation redshifts (cached)
    if cache is not None:
        formation_z = cache.get_formation_redshifts(Snapshot, tree_indices)
    else:
        # Fallback without cache - simplified version
        formation_z = np.zeros(len(tree_indices))

    valid_formation = ~np.isnan(formation_z)
    formation_z_capped = np.clip(formation_z, 0, 1.25)

    # ================================================================
    # Create two-panel figure
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Left panel: scatter plot
    ax1.scatter(log_Mvir, BCG_ICS_fraction, facecolor='dodgerblue', s=75, alpha=0.8, rasterized=True, marker='H', edgecolor='black',
                label='SAGE26 ($M_{vir} > 10^{%.1f}$ M$_\\odot$)' % np.log10(MVIR_THRESHOLD_HALOMASS))
    ax1.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax1.set_ylabel(r'$(M_{\mathrm{BCG}} + M_{\mathrm{ICS}}) / M_{\mathrm{ICS+BCG+Satellites}}$', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10, frameon=False)

    # Right panel: hexbin colored by formation redshift
    right_mask = valid_formation
    hb = ax2.hexbin(log_Mvir[right_mask], BCG_ICS_fraction[right_mask],
                    C=formation_z_capped[right_mask], reduce_C_function=np.mean,
                    gridsize=25, cmap='plasma', mincnt=1, vmin=0, vmax=1.25)
    ax2.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)

    cb = plt.colorbar(hb, ax=ax2)
    cb.set_label(r'$z_{\mathrm{form}}$')

    # Set consistent axis limits
    xlim_min = np.log10(MVIR_THRESHOLD_HALOMASS)
    xlim_max = max(log_Mvir) + 0.2
    ax1.set_xlim(xlim_min, xlim_max)
    ax2.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(0, 1)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_BCG_fraction_vs_halomass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f'  Snapshot: {snap} (z = {z:.2f})')
    print(f'  Centrals plotted: {len(w)} (with BCG, ICS, and satellites > 0)')
    print(f'  Halo mass range: log(Mvir) = {log_Mvir.min():.2f} to {log_Mvir.max():.2f}')
    print(f'  ICS+BCG fraction: median = {np.median(BCG_ICS_fraction):.3f}, '
          f'mean = {np.mean(BCG_ICS_fraction):.3f}')
    print(f'  Formation z: median = {np.nanmedian(formation_z):.2f}, '
          f'mean = {np.nanmean(formation_z):.2f}')
    print(f'  Saved: {outputFile}')


def plot_ics_bcg_fraction_vs_formation_z(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot ICS+BCG fraction vs formation redshift as a scatter plot.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plot
    snap : int, optional
        Snapshot to use. Defaults to latest available (z~0).
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS+BCG Fraction vs Formation Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Use latest snapshot if not specified
    if snap is None:
        snap = sim_params['last_snapshot']

    if snap not in available_snapshots:
        print(f'  Snapshot {snap} not available.')
        return

    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # Read data (using cache if available)
    fields = ['Mvir', 'StellarMass', 'IntraClusterStars', 'Type', 'SAGETreeIndex']
    if cache is not None:
        data = cache.get(Snapshot, fields)
        Mvir = data['Mvir']
        StellarMass = data['StellarMass']
        ICS = data['IntraClusterStars']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']
        # Use cached satellite mass calculation
        satellite_mass = cache.get_satellite_mass(Snapshot)
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        SAGETreeIndex = read_hdf(file_list, Snapshot, 'SAGETreeIndex')
        # Fallback: compute satellite mass without cache
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        satellite_mask = Type != 0
        satellite_central_gids = CentralGalaxyIndex[satellite_mask]
        satellite_masses = StellarMass[satellite_mask]
        insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
        insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
        valid_match = sorted_gids[insert_pos] == satellite_central_gids
        central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
        valid_satellites = central_indices >= 0
        satellite_mass = np.zeros(len(Mvir))
        np.add.at(satellite_mass, central_indices[valid_satellites], satellite_masses[valid_satellites])
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, central_indices[valid_satellites], 1)

    # Select centrals with valid data (same criteria as halo mass plot)
    w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_HALOMASS) &
                 (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w) == 0:
        print('  No galaxies meet selection criteria.')
        return

    # Calculate masses and fraction
    BCG_mass = StellarMass[w]
    ICS_mass = ICS[w]
    Sat_mass = satellite_mass[w]
    tree_indices = SAGETreeIndex[w]

    M_total = BCG_mass + ICS_mass + Sat_mass
    BCG_ICS_fraction = (BCG_mass + ICS_mass) / M_total

    # Get formation redshifts (cached)
    if cache is not None:
        formation_z = cache.get_formation_redshifts(Snapshot, tree_indices)
    else:
        # Fallback without cache - simplified version
        formation_z = np.zeros(len(tree_indices))

    valid_formation = ~np.isnan(formation_z)

    # ================================================================
    # Residual plot: remove halo mass dependence, isolate formation z signal
    # ================================================================
    from scipy.stats import pearsonr

    # Work only with galaxies that have valid formation redshifts
    vf = valid_formation
    log_Mvir = np.log10(Mvir[w][vf])
    log_ICS = np.log10(ICS_mass[vf])
    form_z = formation_z[vf]

    # Fit log(M_ICS) vs log(M_vir) with a linear relation
    fit_coeffs = np.polyfit(log_Mvir, log_ICS, 1)
    log_ICS_predicted = np.polyval(fit_coeffs, log_Mvir)
    residual = log_ICS - log_ICS_predicted  # dex above/below mean relation

    print(f'  M_ICS vs M_vir fit: log(M_ICS) = {fit_coeffs[0]:.3f} * log(M_vir) + {fit_coeffs[1]:.3f}')
    print(f'  Residual scatter: {np.std(residual):.3f} dex')

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(form_z, residual, marker='H', edgecolor='black',
                         c=log_Mvir, cmap='plasma', s=75, alpha=0.8, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=12)

    # Binned median trend
    bin_edges_z = np.linspace(np.min(form_z), np.max(form_z), 10)
    bin_centers_z = 0.5 * (bin_edges_z[:-1] + bin_edges_z[1:])
    median_resid = []
    p16_resid = []
    p84_resid = []
    valid_bins_z = []
    for i in range(len(bin_edges_z) - 1):
        mask = (form_z >= bin_edges_z[i]) & (form_z < bin_edges_z[i + 1])
        if np.sum(mask) >= 5:
            valid_bins_z.append(bin_centers_z[i])
            median_resid.append(np.median(residual[mask]))
            p16_resid.append(np.percentile(residual[mask], 16))
            p84_resid.append(np.percentile(residual[mask], 84))

    if len(valid_bins_z) > 0:
        valid_bins_z = np.array(valid_bins_z)
        median_resid = np.array(median_resid)
        p16_resid = np.array(p16_resid)
        p84_resid = np.array(p84_resid)
        ax.fill_between(valid_bins_z, p16_resid, p84_resid,
                        color='firebrick', alpha=0.2)
        ax.plot(valid_bins_z, median_resid, 'o-', color='firebrick', lw=2.5,
                markersize=7, label='Binned median')

    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)

    ax.set_xlabel(r'Formation Redshift $z_{\mathrm{form}}$', fontsize=14)
    ax.set_ylabel(r'$\Delta \log_{10} M_{\mathrm{ICS}}$ (residual from $M_{\mathrm{vir}}$ fit) [dex]', fontsize=14)
    ax.set_xlim(0, None)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_BCG_fraction_vs_formation_z' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation of residual with formation z
    n_valid = len(form_z)
    if n_valid > 2:
        corr_coef, p_value = pearsonr(form_z, residual)
        print(f'  Residual vs formation z: r = {corr_coef:.3f}, p-value = {p_value:.3e}')
    else:
        print('  Not enough valid data points to calculate correlation.')

    print(f'  Snapshot: {snap} (z = {z:.2f})')
    print(f'  Centrals plotted: {n_valid} (with valid formation redshift)')
    print(f'  Formation z range: {np.min(form_z):.2f} to {np.max(form_z):.2f}')
    print(f'  Saved: {outputFile}')


def plot_ICSmass_vs_halomass(file_list, sim_params, output_dir, cache=None):
    """
    Plot ICS mass vs halo mass for central galaxies with ICS > 0, colored by formation redshift.
    """
    print('\nRunning ICS Mass vs Halo Mass Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']

    # Use latest snapshot for z~0 plot
    snap = sim_params['last_snapshot']
    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # Read data (using cache if available)
    if cache is not None:
        data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'SAGETreeIndex'])
        Mvir = data['Mvir']
        ICS = data['IntraClusterStars']
        StellarMass = data['StellarMass']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        SAGETreeIndex = read_hdf(file_list, Snapshot, 'SAGETreeIndex')
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    # Select centrals with ICS > 0
    w = np.where((Type == 0) & (ICS > 0) & (Mvir >= MVIR_THRESHOLD_HALOMASS) &
                 (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w) == 0:
        print('  No central galaxies with ICS > 0 found.')
        return

    Mvir_sel = Mvir[w]
    ICS_sel = ICS[w]
    tree_indices = SAGETreeIndex[w]

    # Get formation redshifts (cached)
    if cache is not None:
        formation_z = cache.get_formation_redshifts(Snapshot, tree_indices)
    else:
        formation_z = np.zeros(len(tree_indices))

    valid_formation = ~np.isnan(formation_z)
    formation_z_capped = np.clip(formation_z, 0, 1.25)

    # Create scatter plot colored by formation redshift
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(np.log10(Mvir_sel[valid_formation]), np.log10(ICS_sel[valid_formation]),
                    c=formation_z_capped[valid_formation], cmap='plasma', vmin=0, vmax=1.25,
                    s=40, alpha=0.7, rasterized=True)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$z_{\mathrm{form}}$', fontsize=12)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14)

    plt.tight_layout()

    outputFile = os.path.join(output_dir, 'ICS_mass_vs_halomass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate Pearson correlation coefficient
    if np.sum(valid_formation) > 1:
        from scipy.stats import pearsonr
        corr_coef, p_value = pearsonr(np.log10(Mvir_sel[valid_formation]), np.log10(ICS_sel[valid_formation]))
        print(f'  Pearson correlation (Mvir vs ICS): r = {corr_coef:.3f}, p-value = {p_value:.3e}')
    else:
        print('  Not enough valid data points to calculate correlation.')

    # Print statistics
    n_valid = np.sum(valid_formation)
    print(f'  Snapshot: {snap} (z = {z:.2f})')
    print(f'  Centrals plotted: {n_valid}')
    print(f'  ICS range: {np.nanmin(ICS_sel):.2e} to {np.nanmax(ICS_sel):.2e} Msun')
    print(f'  Halo mass range: {np.nanmin(Mvir_sel):.2e} to {np.nanmax(Mvir_sel):.2e} Msun')
    print(f'  Formation z: median = {np.nanmedian(formation_z):.2f}, mean = {np.nanmean(formation_z):.2f}')
    print(f'  Saved: {outputFile}')


def plot_ics_formation_channels(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Track ICS formation channels by analyzing mergeType across all snapshots.

    This function produces three plots:
    1. Cumulative stellar mass processed by each channel (disruption, minor merger, major merger) vs redshift
    2. Count of disruptions and mergers by halo mass bin across all snapshots
    3. ICS fraction vs halo mass, colored by number of disruptions received

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plots
    snap : int, optional
        Snapshot to use for z=0 data. Defaults to latest available.
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Formation Channel Analysis\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Use latest snapshot if not specified
    if snap is None:
        snap = sim_params['last_snapshot']

    # Read FractionDisruptedToICS from HDF5 header
    first_file = file_list[0]
    with h5.File(first_file, 'r') as f:
        try:
            frac_to_ics = float(f['Header/Runtime'].attrs['FractionDisruptedToICS'])
        except KeyError:
            frac_to_ics = 1.0

    print(f'  FractionDisruptedToICS = {frac_to_ics}')

    # ================================================================
    # Step 1: Track contributions by mergeType across snapshots
    # ================================================================
    print('  Tracking mass processed by each formation channel...')

    z_for_plot = []
    disruption_mass = []   # mergeType == 4
    minor_merger_mass = []  # mergeType == 1
    major_merger_mass = []  # mergeType == 2

    for snap_num in sorted(available_snapshots, reverse=True):
        snap_name = f'Snap_{snap_num}'
        z = redshifts[snap_num]
        z_for_plot.append(z)

        # Read data
        fields = ['StellarMass', 'mergeType', 'Mvir']
        data = read_hdf_multi_params(file_list, snap_name, fields)

        StellarMass = data.get('StellarMass', np.array([])) * 1.0e10 / Hubble_h
        mergeType = data.get('mergeType', np.array([]))
        Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h

        if mergeType.size == 0:
            disruption_mass.append(0)
            minor_merger_mass.append(0)
            major_merger_mass.append(0)
            continue

        # Filter by halo mass threshold
        mass_mask = Mvir >= MVIR_THRESHOLD_FORMATION

        # Sum stellar mass by mergeType (only in halos above threshold)
        w_disruption = (mergeType == 4) & mass_mask
        w_minor = (mergeType == 1) & mass_mask
        w_major = (mergeType == 2) & mass_mask

        disruption_mass.append(np.sum(StellarMass[w_disruption]) if np.any(w_disruption) else 0)
        minor_merger_mass.append(np.sum(StellarMass[w_minor]) if np.any(w_minor) else 0)
        major_merger_mass.append(np.sum(StellarMass[w_major]) if np.any(w_major) else 0)

    # Convert to arrays
    z_for_plot = np.array(z_for_plot)
    disruption_mass = np.array(disruption_mass)
    minor_merger_mass = np.array(minor_merger_mass)
    major_merger_mass = np.array(major_merger_mass)

    # ================================================================
    # Plot 1: Cumulative mass processed by each channel
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter to z <= 5
    z_mask = z_for_plot <= 5
    z_plot = z_for_plot[z_mask]
    disrupt_plot = disruption_mass[z_mask]
    minor_plot = minor_merger_mass[z_mask]
    major_plot = major_merger_mass[z_mask]

    # Cumulative sum (going from high z to low z, so reverse before/after)
    # This shows total mass that will be processed from each z down to z=0
    cum_disruption = np.cumsum(disrupt_plot[::-1])[::-1]
    cum_minor = np.cumsum(minor_plot[::-1])[::-1]
    cum_major = np.cumsum(major_plot[::-1])[::-1]

    ax.plot(z_plot, np.log10(cum_disruption + 1), '-', color='firebrick', lw=2, label='Disruption to ICS')
    ax.plot(z_plot, np.log10(cum_minor + 1), '-', color='dodgerblue', lw=2, label='Minor Mergers')
    ax.plot(z_plot, np.log10(cum_major + 1), '-', color='green', lw=2, label='Major Mergers')

    # Calculate dynamic y-axis limits based on data
    all_cum_masses = np.concatenate([cum_disruption, cum_minor, cum_major])
    all_cum_masses = all_cum_masses[all_cum_masses > 0]  # Filter out zeros
    if len(all_cum_masses) > 0:
        y_min = np.floor(np.log10(all_cum_masses.min()))
        y_max = np.ceil(np.log10(all_cum_masses.max()))
    else:
        y_min, y_max = 9, 14

    ax.set_xlabel(r'$z$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}$ Cumulative Stellar Mass Processed $[M_{\odot}]$', fontsize=14)
    ax.set_xlim(0, 2.5)  # z=0 on left, z=5 on right
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_formation_channels_cumulative' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile}')

    # ================================================================
    # Plot 2: Count disruptions/mergers by halo mass bin
    # ================================================================
    print('  Counting disruptions/mergers by halo mass...')

    fig, ax = plt.subplots(figsize=(8, 6))

    bin_edges = np.arange(np.log10(MVIR_THRESHOLD_FORMATION), 15.5, 0.5)
    bin_centers = bin_edges[:-1] + 0.25
    total_disruptions = np.zeros(len(bin_centers))
    total_mergers = np.zeros(len(bin_centers))

    for snap_num in available_snapshots:
        snap_name = f'Snap_{snap_num}'

        fields = ['Mvir', 'mergeType']
        data = read_hdf_multi_params(file_list, snap_name, fields)

        Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h
        mergeType = data.get('mergeType', np.array([]))

        if mergeType.size == 0:
            continue

        log_Mvir = np.log10(np.maximum(Mvir, 1e-10))

        for i in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
            total_disruptions[i] += np.sum(mergeType[mask] == 4)
            total_mergers[i] += np.sum((mergeType[mask] == 1) | (mergeType[mask] == 2))

    # Create bar plot
    width = 0.2
    ax.bar(bin_centers - width/2, total_disruptions, width, label='Disruptions (→ICS)', color='firebrick', alpha=0.7)
    ax.bar(bin_centers + width/2, total_mergers, width, label='Mergers', color='dodgerblue', alpha=0.7)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel('Total Count (all snapshots)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.set_xlim(np.log10(MVIR_THRESHOLD_FORMATION) - 0.1, 15)
    ax.set_yscale('log')

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_formation_mergeType_distribution' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile}')

    # ================================================================
    # Plot 3: ICS fraction colored by number of disruptions received
    # ================================================================
    print('  Counting disruptions per central galaxy...')

    # Count disruptions per central across all snapshots
    # Key: CentralGalaxyIndex at disruption time -> count
    disruption_count = {}

    for snap_num in available_snapshots:
        snap_name = f'Snap_{snap_num}'

        fields = ['mergeType', 'CentralGalaxyIndex']
        data = read_hdf_multi_params(file_list, snap_name, fields)

        mergeType = data.get('mergeType', np.array([]))
        CentralIdx = data.get('CentralGalaxyIndex', np.array([]))

        if mergeType.size == 0:
            continue

        # Find disrupted galaxies and their centrals
        disrupted_mask = mergeType == 4
        central_indices = CentralIdx[disrupted_mask]

        for central_idx in central_indices:
            if central_idx not in disruption_count:
                disruption_count[central_idx] = 0
            disruption_count[central_idx] += 1

    # Get z=0 data
    Snapshot = f'Snap_{snap}'
    fields_z0 = ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'GalaxyIndex']
    if cache is not None:
        data_z0 = cache.get(Snapshot, fields_z0)
        Mvir = data_z0['Mvir']
        ICS = data_z0['IntraClusterStars']
        StellarMass = data_z0['StellarMass']
        Type = data_z0['Type']
        GalaxyIndex = data_z0['GalaxyIndex']
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    # Select centrals with ICS
    w_central = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_FORMATION) & (ICS > 0) &
                         (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w_central) == 0:
        print('  No centrals meet selection criteria.')
        return

    log_Mvir_central = np.log10(Mvir[w_central])
    ICS_frac_central = ICS[w_central] / (BARYON_FRACTION * Mvir[w_central])

    # Get disruption counts for each central
    n_disruptions = np.array([disruption_count.get(GalaxyIndex[i], 0) for i in w_central])

    # Create scatter plot colored by disruption count
    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = np.percentile(n_disruptions[n_disruptions > 0], 95) if np.any(n_disruptions > 0) else 10
    scatter = ax.scatter(log_Mvir_central, ICS_frac_central, c=n_disruptions,
                         cmap='plasma', s=40, alpha=0.7, vmin=0, vmax=vmax, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Disruptions', fontsize=12)

    # Add median line
    bin_width = 0.25
    bin_edges_med = np.arange(np.log10(MVIR_THRESHOLD_FORMATION), 15.5, bin_width)
    bin_centers_med = bin_edges_med[:-1] + bin_width / 2
    median_values = []
    valid_bins = []

    for i in range(len(bin_edges_med) - 1):
        mask = (log_Mvir_central >= bin_edges_med[i]) & (log_Mvir_central < bin_edges_med[i + 1])
        if np.sum(mask) > 2:
            median_values.append(np.median(ICS_frac_central[mask]))
            valid_bins.append(bin_centers_med[i])

    if len(valid_bins) > 0:
        ax.plot(valid_bins, median_values, 'k-', lw=2, label='Median')
        ax.legend(loc='upper left', fontsize=10, frameon=False)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14)
    ax.set_xlim(np.log10(MVIR_THRESHOLD_FORMATION) - 0.1, 15)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_fraction_by_disruption_count' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile}')

    # ================================================================
    # Print statistics
    # ================================================================
    z_final = redshifts[snap]
    print(f'\n  === ICS Formation Channel Statistics ===')
    print(f'  Snapshots analyzed: {len(available_snapshots)}')
    print(f'  Total disruptions (mergeType=4): {int(np.sum(total_disruptions))}')
    print(f'  Total mergers (mergeType=1,2): {int(np.sum(total_mergers))}')
    print(f'\n  At z={z_final:.2f}:')
    print(f'    Centrals analyzed: {len(w_central)}')
    print(f'    Mean disruptions per central: {np.mean(n_disruptions):.1f}')
    print(f'    Median disruptions per central: {np.median(n_disruptions):.0f}')
    print(f'    Max disruptions: {np.max(n_disruptions)}')
    print(f'\n  ICS Formation Channel Analysis completed!')


def plot_ics_channel_breakdown(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot the breakdown of ICS into disruption vs accretion channels using ICS_disrupt and ICS_accrete fields.

    This shows:
    1. ICS_disrupt vs ICS_accrete scatter plot
    2. Fraction from each channel vs halo mass
    3. Stacked bar showing total contributions

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plots
    snap : int, optional
        Snapshot to use. Defaults to latest available.
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Channel Breakdown Analysis (ICS_disrupt vs ICS_accrete)\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']

    # Use latest snapshot if not specified
    if snap is None:
        snap = sim_params['last_snapshot']

    Snapshot = f'Snap_{snap}'
    z = redshifts[snap]

    # Read data - need the new ICS_disrupt and ICS_accrete fields
    fields = ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'ICS_disrupt', 'ICS_accrete',
              'GalaxyIndex', 'CentralGalaxyIndex']

    try:
        data = read_hdf_multi_params(file_list, Snapshot, fields)
        Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h
        ICS = data.get('IntraClusterStars', np.array([])) * 1.0e10 / Hubble_h
        StellarMass = data.get('StellarMass', np.array([])) * 1.0e10 / Hubble_h
        Type = data.get('Type', np.array([]))
        ICS_disrupt = data.get('ICS_disrupt', np.array([])) * 1.0e10 / Hubble_h
        ICS_accrete = data.get('ICS_accrete', np.array([])) * 1.0e10 / Hubble_h
        GalaxyIndex = data.get('GalaxyIndex', np.array([]))
        CentralGalaxyIndex = data.get('CentralGalaxyIndex', np.array([]))
    except Exception as e:
        print(f'  Error reading ICS_disrupt/ICS_accrete fields: {e}')
        print('  These fields may not exist in the output. Skipping this plot.')
        return

    if ICS_disrupt.size == 0 or ICS_accrete.size == 0:
        print('  ICS_disrupt or ICS_accrete fields not found in output. Skipping.')
        return

    # Count satellites per central
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_central_gids = CentralGalaxyIndex[Type != 0]
    ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
    ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
    c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
    n_satellites = np.zeros(len(Type), dtype=int)
    np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    # Select centrals with ICS > 0 and above mass thresholds
    w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_FORMATION) &
                 (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w) == 0:
        print('  No centrals meet selection criteria.')
        return

    log_Mvir = np.log10(Mvir[w])
    ICS_total = ICS[w]
    ICS_disruption = ICS_disrupt[w]
    ICS_accreteretion = ICS_accrete[w]

    # Sanity check: ICS should equal ICS_disrupt + ICS_accrete
    ICS_sum = ICS_disruption + ICS_accreteretion
    residual = ICS_total - ICS_sum
    frac_residual = np.abs(residual) / (ICS_total + 1e-10)

    print(f'  Snapshot {snap} (z = {z:.2f})')
    print(f'  Centrals with ICS > 0: {len(w)}')

    # Report any significant discrepancies
    large_residual = frac_residual > 0.01  # More than 1% difference
    if np.any(large_residual):
        n_bad = np.sum(large_residual)
        print(f'  WARNING: {n_bad} galaxies have ICS != ICS_disrupt + ICS_accrete (>1% difference)')
        print(f'    Max fractional residual: {np.max(frac_residual):.3f}')
    else:
        print(f'  Sanity check PASSED: ICS = ICS_disrupt + ICS_accrete for all galaxies')

    # Calculate fractions
    frac_disc = ICS_disruption / ICS_total
    frac_acc = ICS_accreteretion / ICS_total

    print(f'  Mean fraction from disruption: {np.mean(frac_disc):.3f}')
    print(f'  Mean fraction from accretion: {np.mean(frac_acc):.3f}')

    # ================================================================
    # Plot 1: ICS_disrupt vs ICS_accrete scatter plot (exclude zeros)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    # Only plot points where both channels are > 0
    both_nonzero = (ICS_disruption > 0) & (ICS_accreteretion > 0)
    if np.sum(both_nonzero) == 0:
        print('  No galaxies with both ICS_disrupt > 0 and ICS_accrete > 0. Skipping ICS_disrupt_vs_acc plot.')
        plt.close()
    else:
        # Color by halo mass
        scatter = ax.scatter(np.log10(ICS_disruption[both_nonzero]), np.log10(ICS_accreteretion[both_nonzero]),
                             c=log_Mvir[both_nonzero], cmap='plasma', s=40, alpha=0.7, rasterized=True)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=12)

        # Add 1:1 line
        lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')

        ax.set_xlabel(r'$\log_{10} m_{\mathrm{ICS,disc}}\ [M_{\odot}]$ (from disruptions)', fontsize=12)
        ax.set_ylabel(r'$\log_{10} m_{\mathrm{ICS,acc}}\ [M_{\odot}]$ (from accretion)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10, frameon=False)

        plt.tight_layout()
        outputFile = os.path.join(output_dir, 'ICS_disrupt_vs_acc' + OutputFormat)
        plt.savefig(outputFile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {outputFile}')

    # ================================================================
    # Plot 2: Fraction from each channel vs halo mass (median lines with shading)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Bin by halo mass
    bin_width = 0.25
    bin_edges = np.arange(np.log10(MVIR_THRESHOLD_FORMATION) - bin_width / 2, 15.5, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Compute median and percentiles for each bin
    median_disc = []
    p16_disc = []
    p84_disc = []
    median_acc = []
    p16_acc = []
    p84_acc = []
    valid_bins = []

    for i in range(len(bin_edges) - 1):
        mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
        if np.sum(mask) >= 5:  # Require at least 5 galaxies per bin
            valid_bins.append(bin_centers[i])
            median_disc.append(np.median(frac_disc[mask]))
            p16_disc.append(np.percentile(frac_disc[mask], 16))
            p84_disc.append(np.percentile(frac_disc[mask], 84))
            median_acc.append(np.median(frac_acc[mask]))
            p16_acc.append(np.percentile(frac_acc[mask], 16))
            p84_acc.append(np.percentile(frac_acc[mask], 84))

    valid_bins = np.array(valid_bins)
    median_disc = np.array(median_disc)
    p16_disc = np.array(p16_disc)
    p84_disc = np.array(p84_disc)
    median_acc = np.array(median_acc)
    p16_acc = np.array(p16_acc)
    p84_acc = np.array(p84_acc)

    # Disruption channel (dodgerblue)
    ax.fill_between(valid_bins, p16_disc, p84_disc,
                    color='gray', alpha=0.3, label=None)
    ax.plot(valid_bins, median_disc, color='k', lw=2.5, linestyle='--',
            label=r'Disruption')

    # Accretion channel (red)
    ax.fill_between(valid_bins, p16_acc, p84_acc,
                    color='gray', alpha=0.3, label=None)
    ax.plot(valid_bins, median_acc, color='k', linestyle='-', lw=2.5,
            label=r'Accretion')

    ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'Fraction of ICS from channel', fontsize=14)
    ax.set_xlim(np.log10(MVIR_THRESHOLD_FORMATION) - 0.1, 15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=11, frameon=False)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_channel_fraction_vs_Mvir' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile}')

    # ================================================================
    # Print summary statistics
    # ================================================================
    total_ICS_disrupt = np.sum(ICS_disrupt)
    total_ICS_accrete = np.sum(ICS_accrete)
    total_ICS = np.sum(ICS_total)

    print(f'\n  === ICS Channel Breakdown Summary ===')
    print(f'  Total ICS mass: {total_ICS:.2e} Msun')
    print(f'  From disruptions: {total_ICS_disrupt:.2e} Msun ({100*total_ICS_disrupt/total_ICS:.1f}%)')
    print(f'  From accretion: {total_ICS_accrete:.2e} Msun ({100*total_ICS_accrete/total_ICS:.1f}%)')
    print(f'\n  ICS Channel Breakdown Analysis completed!')


def plot_ics_channel_fraction_grid(file_list, sim_params, output_dir, cache=None):
    """
    Plot a 2x4 grid of ICS channel fraction vs Mvir at different redshifts.

    This shows how the crossover point between disruption and accretion channels
    evolves with redshift.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plots
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Channel Fraction vs Mvir Grid (multiple redshifts)\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']

    # Find closest snapshots to target redshifts
    target_snaps = []
    actual_redshifts = []
    for z_target in TARGET_REDSHIFTS:
        z_diff = np.abs(np.array(redshifts) - z_target)
        snap = np.argmin(z_diff)
        target_snaps.append(snap)
        actual_redshifts.append(redshifts[snap])

    print(f'  Target redshifts: {TARGET_REDSHIFTS}')
    print(f'  Actual redshifts: {[f"{z:.2f}" for z in actual_redshifts]}')

    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fields = ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'ICS_disrupt', 'ICS_accrete',
              'GalaxyIndex', 'CentralGalaxyIndex']

    for i, (snap, z) in enumerate(zip(target_snaps, actual_redshifts)):
        ax = axes[i]
        Snapshot = f'Snap_{snap}'

        try:
            data = read_hdf_multi_params(file_list, Snapshot, fields)
            Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h
            ICS = data.get('IntraClusterStars', np.array([])) * 1.0e10 / Hubble_h
            StellarMass = data.get('StellarMass', np.array([])) * 1.0e10 / Hubble_h
            Type = data.get('Type', np.array([]))
            ICS_disrupt = data.get('ICS_disrupt', np.array([])) * 1.0e10 / Hubble_h
            ICS_accrete = data.get('ICS_accrete', np.array([])) * 1.0e10 / Hubble_h
            GalaxyIndex = data.get('GalaxyIndex', np.array([]))
            CentralGalaxyIndex = data.get('CentralGalaxyIndex', np.array([]))
        except Exception as e:
            print(f'  Error reading data for snap {snap}: {e}')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'z = {z:.1f}', fontsize=12)
            continue

        if ICS_disrupt.size == 0 or ICS_accrete.size == 0:
            ax.text(0.5, 0.5, 'No ICS channel data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'z = {z:.1f}', fontsize=12)
            continue

        # Count satellites per central
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with ICS > 0 and above thresholds
        w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_MASSFUNCTION) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))[0]

        if len(w) < 5:
            ax.text(0.5, 0.5, f'N = {len(w)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'z = {z:.1f}', fontsize=12)
            continue

        log_Mvir = np.log10(Mvir[w])
        ICS_total = ICS[w]
        ICS_disruption = ICS_disrupt[w]
        ICS_accreteretion = ICS_accrete[w]

        # Calculate fractions
        frac_disc = ICS_disruption / ICS_total
        frac_acc = ICS_accreteretion / ICS_total

        # Bin by halo mass
        bin_width = 0.25
        bin_edges = np.arange(np.log10(MVIR_THRESHOLD_MASSFUNCTION), 15.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        # Compute median and percentiles for each bin
        median_disc = []
        p16_disc = []
        p84_disc = []
        median_acc = []
        p16_acc = []
        p84_acc = []
        valid_bins = []

        for j in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[j]) & (log_Mvir < bin_edges[j + 1])
            if np.sum(mask) >= 5:  # Require at least 5 galaxies per bin
                valid_bins.append(bin_centers[j])
                median_disc.append(np.median(frac_disc[mask]))
                p16_disc.append(np.percentile(frac_disc[mask], 16))
                p84_disc.append(np.percentile(frac_disc[mask], 84))
                median_acc.append(np.median(frac_acc[mask]))
                p16_acc.append(np.percentile(frac_acc[mask], 16))
                p84_acc.append(np.percentile(frac_acc[mask], 84))

        if len(valid_bins) == 0:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'z = {z:.1f}', fontsize=12)
            continue

        valid_bins = np.array(valid_bins)
        median_disc = np.array(median_disc)
        p16_disc = np.array(p16_disc)
        p84_disc = np.array(p84_disc)
        median_acc = np.array(median_acc)
        p16_acc = np.array(p16_acc)
        p84_acc = np.array(p84_acc)

        # Disruption channel (dodgerblue)
        ax.fill_between(valid_bins, p16_disc, p84_disc,
                        color='gray', alpha=0.3)
        ax.plot(valid_bins, median_disc, '--', color='black', lw=2,
                markersize=4, label='Disruption')

        # Accretion channel (red)
        ax.fill_between(valid_bins, p16_acc, p84_acc,
                        color='gray', alpha=0.3)
        ax.plot(valid_bins, median_acc, '-', color='black', lw=2,
                markersize=4, label='Accretion')

        ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5)

        # Find and mark crossover point
        if len(median_disc) > 1 and len(median_acc) > 1:
            diff = median_disc - median_acc
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_changes) > 0:
                # Linear interpolation to find crossover mass
                idx = sign_changes[0]
                if idx + 1 < len(valid_bins):
                    x1, x2 = valid_bins[idx], valid_bins[idx + 1]
                    y1, y2 = diff[idx], diff[idx + 1]
                    crossover_mass = x1 - y1 * (x2 - x1) / (y2 - y1)
                    ax.axvline(crossover_mass, color='green', linestyle=':', lw=1.5, alpha=0.8)
                    ax.text(crossover_mass + 0.05, 0.95, f'{crossover_mass:.1f}',
                            fontsize=8, color='green', va='top', transform=ax.get_xaxis_transform())

        ax.set_title(f'z = {z:.1f}', fontsize=11)
        ax.set_xlim(np.log10(MVIR_THRESHOLD_MASSFUNCTION) - 0.1, 15)
        ax.set_ylim(-0.05, 1.05)

        if i == 0:
            ax.legend(loc='center left', fontsize=8, frameon=False)

        print(f'  z = {z:.2f}: {len(w)} centrals, {len(valid_bins)} valid bins')

    # Common axis labels
    fig.text(0.5, 0.02, r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', ha='center', fontsize=14)
    fig.text(0.02, 0.5, r'Fraction of ICS from channel', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    outputFile = os.path.join(output_dir, 'ICS_channel_fraction_vs_Mvir_grid' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {outputFile}')
    print(f'  ICS Channel Fraction Grid completed!')


def plot_icl_contribution_by_infall_mass(file_list, sim_params, output_dir, cache=None):
    """
    Plot the fractional contribution to ICL as a function of infall stellar mass.

    Implements the methodology from Bradley et al.:
    f(M∗) = M∗ × f_lib(M∗) × Φ(M∗) / ∫ M∗ × f_lib(M∗) × Φ(M∗) · dM∗

    Where:
    - Φ(M∗) is the infall stellar mass function (histogram of processed satellites)
    - f_lib(M∗) is the liberated fraction (mass contributed to ICL / infall mass)

    The result is normalized per dex in infall stellar mass.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plots
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    from scipy.optimize import curve_fit

    print('\nRunning ICL Contribution by Infall Stellar Mass\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    first_snap = sim_params['first_snapshot']
    last_snap = sim_params['last_snapshot']

    # Track disruption and merger events across snapshots
    # mergeType == 4 indicates disruption to ICS
    # mergeType == 1 or 2 indicates merger (minor/major) to BCG
    disrupt_infall_masses = []  # Stellar mass at infall for disrupted satellites
    disrupt_contributed_masses = []  # Stellar mass contributed to ICL (at disruption)

    merger_infall_masses = []  # Stellar mass at infall for merged satellites
    merger_contributed_masses = []  # Stellar mass contributed to BCG (at merger)

    fields = ['Mvir', 'StellarMass', 'Type', 'CentralGalaxyIndex', 'GalaxyIndex',
              'infallStellarMass', 'mergeType']

    print(f'  Scanning snapshots {first_snap} to {last_snap} for disruption and merger events...')

    for snap in range(first_snap, last_snap + 1):
        Snapshot = f'Snap_{snap}'

        try:
            data = read_hdf_multi_params(file_list, Snapshot, fields)
            Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h
            StellarMass = data.get('StellarMass', np.array([])) * 1.0e10 / Hubble_h
            Type = data.get('Type', np.array([]))
            CentralGalaxyIndex = data.get('CentralGalaxyIndex', np.array([]))
            GalaxyIndex = data.get('GalaxyIndex', np.array([]))
            infallStellarMass = data.get('infallStellarMass', np.array([])) * 1.0e10 / Hubble_h
            mergeType = data.get('mergeType', np.array([]))
        except Exception as e:
            continue

        if infallStellarMass.size == 0 or mergeType.size == 0:
            continue

        # Find satellites that were disrupted to ICS (mergeType == 4)
        disrupted_mask = (mergeType == 4) & (infallStellarMass > 0)
        disrupted_indices = np.where(disrupted_mask)[0]

        # Find satellites that merged with BCG (mergeType == 1 or 2)
        merged_mask = ((mergeType == 1) | (mergeType == 2)) & (infallStellarMass > 0)
        merged_indices = np.where(merged_mask)[0]

        # Process disruptions (contribute to ICL)
        for idx in disrupted_indices:
            central_idx = CentralGalaxyIndex[idx]
            central_match = np.where(GalaxyIndex == central_idx)[0]
            if len(central_match) == 0:
                continue
            central_mvir = Mvir[central_match[0]]
            if central_mvir < MVIR_THRESHOLD_FORMATION:
                continue

            disrupt_infall_masses.append(infallStellarMass[idx])
            disrupt_contributed_masses.append(StellarMass[idx])

        # Process mergers (contribute to BCG)
        for idx in merged_indices:
            central_idx = CentralGalaxyIndex[idx]
            central_match = np.where(GalaxyIndex == central_idx)[0]
            if len(central_match) == 0:
                continue
            central_mvir = Mvir[central_match[0]]
            if central_mvir < MVIR_THRESHOLD_FORMATION:
                continue

            merger_infall_masses.append(infallStellarMass[idx])
            merger_contributed_masses.append(StellarMass[idx])

    disrupt_infall_masses = np.array(disrupt_infall_masses)
    disrupt_contributed_masses = np.array(disrupt_contributed_masses)
    merger_infall_masses = np.array(merger_infall_masses)
    merger_contributed_masses = np.array(merger_contributed_masses)

    if len(disrupt_infall_masses) == 0:
        print('  No disruption events found with infallStellarMass > 0.')
        print('  Please ensure SAGE was run with the updated code that tracks infallStellarMass.')
        return

    print(f'  Disruption events: {len(disrupt_infall_masses)}')
    print(f'  Merger events: {len(merger_infall_masses)}')
    print(f'  Total mass to ICL: {np.sum(disrupt_contributed_masses):.2e} Msun')
    print(f'  Total mass to BCG (mergers): {np.sum(merger_contributed_masses):.2e} Msun')

    # Combine all processed satellites for Φ(M∗) calculation
    all_infall_masses = np.concatenate([disrupt_infall_masses, merger_infall_masses])

    # Define stellar mass bins
    bin_width = 0.25  # dex
    log_mass_min = 8.0
    log_mass_max = 12.5
    # For fitting, only use M* above resolution limit
    fit_mass_min = np.log10(STELLARMASS_THRESHOLD)

    bin_edges = np.arange(log_mass_min, log_mass_max + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    log_all_infall = np.log10(all_infall_masses)
    log_disrupt_infall = np.log10(disrupt_infall_masses)
    log_merger_infall = np.log10(merger_infall_masses) if len(merger_infall_masses) > 0 else np.array([])

    # =========================================================================
    # 1. Compute Φ(M∗) - infall stellar mass function (counts per bin)
    # =========================================================================
    phi_counts, _ = np.histogram(log_all_infall, bins=bin_edges)
    phi = phi_counts.astype(float)

    # Fit Schechter function to Φ(M∗) for M* > 10^9
    def schechter_log(log_m, phi_n, log_m_k, alpha):
        """Schechter function in log space: Φ(M) = Φ_n * (M/M_k)^α * exp(-M/M_k)"""
        m = 10**log_m
        m_k = 10**log_m_k
        return phi_n * (m / m_k)**alpha * np.exp(-m / m_k)

    # Only fit bins above resolution limit
    fit_mask = bin_centers >= fit_mass_min
    phi_to_fit = phi[fit_mask]
    centers_to_fit = bin_centers[fit_mask]

    # Initial guesses for Schechter parameters
    try:
        # Filter out zero counts for fitting
        nonzero_mask = phi_to_fit > 0
        if np.sum(nonzero_mask) >= 3:
            popt_phi, _ = curve_fit(
                schechter_log,
                centers_to_fit[nonzero_mask],
                phi_to_fit[nonzero_mask],
                p0=[np.max(phi_to_fit), 10.5, -1.0],
                bounds=([0, 9, -3], [1e6, 12, 1]),
                maxfev=5000
            )
            phi_n_fit, log_m_k_fit, alpha_fit = popt_phi
            phi_fit = schechter_log(bin_centers, phi_n_fit, log_m_k_fit, alpha_fit)
            schechter_fit_success = True
            print(f'  Schechter fit: Φ_n={phi_n_fit:.1f}, log(M_k)={log_m_k_fit:.2f}, α={alpha_fit:.2f}')
        else:
            schechter_fit_success = False
    except Exception as e:
        print(f'  Warning: Schechter fit failed: {e}')
        schechter_fit_success = False

    # =========================================================================
    # 2. Compute f_lib(M∗) - liberated fraction per bin
    #    f_lib = (mass contributed to ICL) / (total infall mass) for each bin
    # =========================================================================
    f_lib = np.zeros(len(bin_centers))

    for i in range(len(bin_edges) - 1):
        # Total infall mass of all processed satellites in this bin
        mask_all = (log_all_infall >= bin_edges[i]) & (log_all_infall < bin_edges[i + 1])
        total_infall_in_bin = np.sum(all_infall_masses[mask_all])

        # Mass contributed to ICL by disrupted satellites in this bin
        mask_disrupt = (log_disrupt_infall >= bin_edges[i]) & (log_disrupt_infall < bin_edges[i + 1])
        icl_contributed_in_bin = np.sum(disrupt_contributed_masses[mask_disrupt])

        if total_infall_in_bin > 0:
            f_lib[i] = icl_contributed_in_bin / total_infall_in_bin

    # =========================================================================
    # 3. Compute f(M∗) = M∗ × f_lib(M∗) × Φ(M∗) / ∫ M∗ × f_lib(M∗) × Φ(M∗) dM∗
    #    Both ICL and BCG+ICL normalized by the SAME integral (BCG+ICL total)
    #    so that ICL appears as a subset of BCG+ICL
    # =========================================================================
    M_star = 10**bin_centers  # Linear stellar mass at bin centers

    # First compute f_lib for BCG+ICL (we need this for the normalization)
    f_lib_bcg_icl = np.zeros(len(bin_centers))

    for i in range(len(bin_edges) - 1):
        mask_all = (log_all_infall >= bin_edges[i]) & (log_all_infall < bin_edges[i + 1])
        total_infall_in_bin = np.sum(all_infall_masses[mask_all])

        # ICL contribution
        mask_disrupt = (log_disrupt_infall >= bin_edges[i]) & (log_disrupt_infall < bin_edges[i + 1])
        icl_contrib = np.sum(disrupt_contributed_masses[mask_disrupt])

        # BCG contribution (mergers)
        if len(log_merger_infall) > 0:
            mask_merger = (log_merger_infall >= bin_edges[i]) & (log_merger_infall < bin_edges[i + 1])
            bcg_contrib = np.sum(merger_contributed_masses[mask_merger])
        else:
            bcg_contrib = 0

        if total_infall_in_bin > 0:
            f_lib_bcg_icl[i] = (icl_contrib + bcg_contrib) / total_infall_in_bin

    # Compute integrands
    integrand_icl = M_star * f_lib * phi
    integrand_bcg_icl = M_star * f_lib_bcg_icl * phi

    # Use BCG+ICL integral for normalizing BOTH curves
    integral_bcg_icl = np.sum(integrand_bcg_icl) * bin_width

    if integral_bcg_icl > 0:
        # Both normalized by the same integral so ICL is a subset of BCG+ICL
        f_M_icl = integrand_icl / integral_bcg_icl
        f_M_bcg_icl = integrand_bcg_icl / integral_bcg_icl
    else:
        f_M_icl = np.zeros_like(integrand_icl)
        f_M_bcg_icl = np.zeros_like(integrand_bcg_icl)

    # If Schechter fit succeeded, compute extrapolated version
    if schechter_fit_success:
        integrand_icl_fit = M_star * f_lib * phi_fit
        integrand_bcg_icl_fit = M_star * f_lib_bcg_icl * phi_fit

        # Use BCG+ICL fit integral for normalizing both
        integral_bcg_icl_fit = np.sum(integrand_bcg_icl_fit) * bin_width
        if integral_bcg_icl_fit > 0:
            f_M_icl_fit = integrand_icl_fit / integral_bcg_icl_fit
            f_M_bcg_icl_fit = integrand_bcg_icl_fit / integral_bcg_icl_fit
        else:
            f_M_icl_fit = np.zeros_like(integrand_icl_fit)
            f_M_bcg_icl_fit = np.zeros_like(integrand_bcg_icl_fit)

    # =========================================================================
    # Bootstrap resampling for error estimation
    # =========================================================================
    n_bootstrap = 100
    n_disrupt = len(disrupt_infall_masses)
    n_merger = len(merger_infall_masses)

    f_M_icl_bootstrap = np.zeros((n_bootstrap, len(bin_centers)))
    f_M_bcg_icl_bootstrap = np.zeros((n_bootstrap, len(bin_centers)))

    for b in range(n_bootstrap):
        # Resample disruptions with replacement
        disrupt_idx = np.random.choice(n_disrupt, size=n_disrupt, replace=True)
        boot_disrupt_infall = disrupt_infall_masses[disrupt_idx]
        boot_disrupt_contrib = disrupt_contributed_masses[disrupt_idx]

        # Resample mergers with replacement
        if n_merger > 0:
            merger_idx = np.random.choice(n_merger, size=n_merger, replace=True)
            boot_merger_infall = merger_infall_masses[merger_idx]
            boot_merger_contrib = merger_contributed_masses[merger_idx]
        else:
            boot_merger_infall = np.array([])
            boot_merger_contrib = np.array([])

        # Combine for Φ(M∗)
        boot_all_infall = np.concatenate([boot_disrupt_infall, boot_merger_infall])
        boot_log_all_infall = np.log10(boot_all_infall)
        boot_log_disrupt_infall = np.log10(boot_disrupt_infall)
        boot_log_merger_infall = np.log10(boot_merger_infall) if len(boot_merger_infall) > 0 else np.array([])

        # Compute Φ(M∗) for this bootstrap sample
        boot_phi_counts, _ = np.histogram(boot_log_all_infall, bins=bin_edges)
        boot_phi = boot_phi_counts.astype(float)

        # Compute f_lib for ICL
        boot_f_lib = np.zeros(len(bin_centers))
        for i in range(len(bin_edges) - 1):
            mask_all = (boot_log_all_infall >= bin_edges[i]) & (boot_log_all_infall < bin_edges[i + 1])
            total_infall_in_bin = np.sum(boot_all_infall[mask_all])

            mask_disrupt = (boot_log_disrupt_infall >= bin_edges[i]) & (boot_log_disrupt_infall < bin_edges[i + 1])
            icl_contributed_in_bin = np.sum(boot_disrupt_contrib[mask_disrupt])

            if total_infall_in_bin > 0:
                boot_f_lib[i] = icl_contributed_in_bin / total_infall_in_bin

        # Compute f_lib for BCG+ICL
        boot_f_lib_bcg_icl = np.zeros(len(bin_centers))
        for i in range(len(bin_edges) - 1):
            mask_all = (boot_log_all_infall >= bin_edges[i]) & (boot_log_all_infall < bin_edges[i + 1])
            total_infall_in_bin = np.sum(boot_all_infall[mask_all])

            mask_disrupt = (boot_log_disrupt_infall >= bin_edges[i]) & (boot_log_disrupt_infall < bin_edges[i + 1])
            icl_contrib = np.sum(boot_disrupt_contrib[mask_disrupt])

            if len(boot_log_merger_infall) > 0:
                mask_merger = (boot_log_merger_infall >= bin_edges[i]) & (boot_log_merger_infall < bin_edges[i + 1])
                bcg_contrib = np.sum(boot_merger_contrib[mask_merger])
            else:
                bcg_contrib = 0

            if total_infall_in_bin > 0:
                boot_f_lib_bcg_icl[i] = (icl_contrib + bcg_contrib) / total_infall_in_bin

        # Compute f(M∗) for this bootstrap sample
        boot_M_star = 10**bin_centers
        boot_integrand_icl = boot_M_star * boot_f_lib * boot_phi
        boot_integrand_bcg_icl = boot_M_star * boot_f_lib_bcg_icl * boot_phi

        boot_integral_bcg_icl = np.sum(boot_integrand_bcg_icl) * bin_width

        if boot_integral_bcg_icl > 0:
            f_M_icl_bootstrap[b, :] = boot_integrand_icl / boot_integral_bcg_icl
            f_M_bcg_icl_bootstrap[b, :] = boot_integrand_bcg_icl / boot_integral_bcg_icl

    # Compute 16th and 84th percentiles (1-sigma equivalent)
    f_M_icl_lo = np.percentile(f_M_icl_bootstrap, 16, axis=0)
    f_M_icl_hi = np.percentile(f_M_icl_bootstrap, 84, axis=0)
    f_M_bcg_icl_lo = np.percentile(f_M_bcg_icl_bootstrap, 16, axis=0)
    f_M_bcg_icl_hi = np.percentile(f_M_bcg_icl_bootstrap, 84, axis=0)

    # =========================================================================
    # Create the plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask for bins with data vs extrapolation
    data_mask = phi > 0
    extrap_mask = (bin_centers < fit_mass_min) & (phi > 0)
    fit_region_mask = bin_centers >= fit_mass_min

    # Plot ICL contribution (solid line for data region) with bootstrap errors
    ax.fill_between(bin_centers[data_mask], f_M_icl_lo[data_mask], f_M_icl_hi[data_mask],
                    color='k', alpha=0.2)
    ax.plot(bin_centers[data_mask], f_M_icl[data_mask], color='k', lw=2.5,
            label='ICL')

    # Plot BCG+ICL contribution (fainter line) with bootstrap errors
    ax.fill_between(bin_centers[data_mask], f_M_bcg_icl_lo[data_mask], f_M_bcg_icl_hi[data_mask],
                    color='k', alpha=0.1)
    ax.plot(bin_centers[data_mask], f_M_bcg_icl[data_mask], '--', color='k',
            lw=1.5, alpha=0.5, label='BCG + ICL')

    # If Schechter fit succeeded, show extrapolation with dotted lines
    if schechter_fit_success:
        # Extrapolation region (below fit range)
        extrap_region = bin_centers < fit_mass_min
        if np.any(extrap_region):
            ax.plot(bin_centers[extrap_region], f_M_icl_fit[extrap_region], ':',
                    color='k', lw=2, alpha=0.7)
            ax.plot(bin_centers[extrap_region], f_M_bcg_icl_fit[extrap_region], ':',
                    color='k', lw=1.5, alpha=0.4)

    # Find and mark the peak for ICL
    if np.any(f_M_icl > 0):
        peak_idx = np.argmax(f_M_icl)
        peak_mass = bin_centers[peak_idx]
        peak_frac = f_M_icl[peak_idx]
        ax.axvline(peak_mass, color='firebrick', linestyle=':', lw=1.5, alpha=0.6)
        ax.text(peak_mass + 0.3, peak_frac * 0.95, f'Peak: $10^{{{peak_mass:.1f}}}$ M$_\\odot$',
                fontsize=10, color='firebrick', va='top')

    # Mark resolution limit
    ax.axvline(fit_mass_min, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.text(fit_mass_min - 0.1, ax.get_ylim()[1] * 0.9, 'Resolution\nlimit',
            fontsize=9, color='gray', ha='right', va='top')

    ax.set_xlabel(r'$\log_{10} m_{*,\mathrm{infall}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'Fraction of stellar mass contributed to ICL', fontsize=14)
    ax.set_xlim(log_mass_min - 0.1, log_mass_max + 0.1)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=11, frameon=False)

    # # Add info text
    # total_icl = np.sum(disrupt_contributed_masses)
    # info_text = (f'N$_{{disruptions}}$ = {len(disrupt_infall_masses)}\n'
    #              f'N$_{{mergers}}$ = {len(merger_infall_masses)}\n'
    #              f'M$_{{ICL}}$ = {total_icl:.2e} M$_\\odot$')
    # ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10,
    #         va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICL_contribution_by_infall_mass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n  Peak ICL contribution at log(M*,infall) = {peak_mass:.2f}')
    print(f'  Saved: {outputFile}')


def plot_ics_vs_bcg_mass(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot ICS mass vs BCG mass for central galaxies with ICS > 0, colored by formation redshift.
    """
    print('\nRunning ICS Mass vs BCG Mass Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']

    # Use latest snapshot for z~0 plot
    snap = snap if snap is not None else sim_params['last_snapshot']
    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # Read data (using cache if available)
    if cache is not None:
        data = cache.get(Snapshot, ['Mvir', 'StellarMass', 'IntraClusterStars', 'Type', 'SAGETreeIndex'])
        StellarMass = data['StellarMass']
        Mvir = data['Mvir']
        ICS = data['IntraClusterStars']
        Type = data['Type']
        SAGETreeIndex = data['SAGETreeIndex']
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(file_list, Snapshot, 'Type')
        SAGETreeIndex = read_hdf(file_list, Snapshot, 'SAGETreeIndex')
        GalaxyIndex = read_hdf(file_list, Snapshot, 'GalaxyIndex')
        CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    # Select centrals with ICS > 0
    w = np.where((Type == 0) & (ICS > 0) & (StellarMass > STELLARMASS_THRESHOLD) & (Mvir >= MVIR_THRESHOLD_HALOMASS) & (n_satellites >= MIN_SATELLITES))[0]

    if len(w) == 0:
        print('  No central galaxies with ICS > 0 found.')
        return
    
    BCG_mass = StellarMass[w]
    ICS_mass = ICS[w]
    tree_indices = SAGETreeIndex[w]

    # Plot ICS mass vs BCG mass colored by formation redshift

    # Get formation redshifts (cached)
    if cache is not None:
        formation_z = cache.get_formation_redshifts(Snapshot, tree_indices)
    else:
        formation_z = np.zeros(len(tree_indices))

    plt.figure(figsize=(10, 7))

    # Cap formation redshift for color scale
    formation_z_capped = np.clip(formation_z, 0, 1.25)

    sc = plt.scatter(np.log10(BCG_mass), np.log10(ICS_mass), c=formation_z_capped, s=75, alpha=0.8, edgecolor='black', rasterized=True, marker='H',
                     label=f'SAGE26 ($M_{{vir}} > 10^{{{np.log10(MVIR_THRESHOLD_HALOMASS):.1f}}}$ M$_\\odot$)', cmap='plasma', vmin=0, vmax=1.25)
    cb = plt.colorbar(sc)
    cb.set_label(r'Formation Redshift $z_{\mathrm{form}}$', fontsize=12)
    plt.xlabel(r'$\log_{10} M_{\mathrm{BCG}}\ [M_{\odot}]$', fontsize=14)
    plt.ylabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14)
    plt.legend(loc='upper left', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_mass_vs_BCG_mass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate Pearson correlation coefficient
    if np.sum(~np.isnan(formation_z)) > 1:
        from scipy.stats import pearsonr
        corr_coef, p_value = pearsonr(np.log10(BCG_mass[~np.isnan(formation_z)]), np.log10(ICS_mass[~np.isnan(formation_z)]))
        print(f'  Pearson correlation (BCG mass vs ICS mass): r = {corr_coef:.3f}, p-value = {p_value:.3e}')
    else:
        print('  Not enough valid data points to calculate correlation.')

    # Print statistics
    print(f'  Snapshot: {snap} (z = {z:.2f})')
    print(f'  Centrals plotted: {len(w)} (with BCG mass > {STELLARMASS_THRESHOLD:.2e} Msun and ICS > 0)')
    print(f'  BCG mass range: {np.min(BCG_mass):.2e} to {np.max(BCG_mass):.2e} Msun')
    print(f'  ICS mass range: {np.min(ICS_mass):.2e} to {np.max(ICS_mass):.2e} Msun')
    print(f'  Formation z: median = {np.nanmedian(formation_z):.2f}, mean = {np.nanmean(formation_z):.2f}')
    print(f'  Saved: {outputFile}')


def plot_ICS_mass_vs_mvir_by_channel(file_list, sim_params, output_dir, snap=None, cache=None):
    """
    Plot M_ICS vs M_vir at z=0 coloured by ICS formation channel.

    Reads ICS_disrupt and ICS_accrete directly from the SAGE HDF5 output — these are
    cumulative channel reservoirs maintained by the C model:

      ICS_disrupt  = total ICS contributed by tidal disruption of pure field
                  satellites (satellites that carried no prior ICS).
      ICS_accrete   = total ICS accreted from former group centrals (satellites
                  that already carried their own ICS when disrupted).

    f_accretion = ICS_accrete / (ICS_disrupt + ICS_accrete)

    Two outputs are saved:
      1. log M_ICS_total vs log M_vir coloured by f_accretion
      2. log M_ICS_disruption vs log M_ICS_accreteretion coloured by log M_vir
    """
    print('\nRunning M_ICS vs M_vir by Formation Channel Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']

    snap = snap if snap is not None else sim_params['last_snapshot']
    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    # ------------------------------------------------------------------
    # Read z=0 central galaxies
    # ------------------------------------------------------------------
    fields_z0 = ['Mvir', 'IntraClusterStars', 'StellarMass', 'ICS_disrupt', 'ICS_accrete', 'Type',
                  'GalaxyIndex', 'CentralGalaxyIndex']
    if cache is not None:
        data = cache.get(Snapshot, fields_z0)
        n_satellites = cache.get_satellite_count(Snapshot)
    else:
        data = read_hdf_multi_params(file_list, Snapshot, fields_z0)
        for f in ('Mvir', 'IntraClusterStars', 'StellarMass', 'ICS_disrupt', 'ICS_accrete'):
            if f in data:
                data[f] = data[f] * 1.0e10 / Hubble_h
        GalaxyIndex = data['GalaxyIndex']
        CentralGalaxyIndex = data['CentralGalaxyIndex']
        Type_tmp = data['Type']
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type_tmp != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type_tmp), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

    Mvir        = data['Mvir']
    ICS_z0      = data['IntraClusterStars']
    StellarMass = data['StellarMass']
    ICS_d_all   = data['ICS_disrupt']
    ICS_a_all   = data['ICS_accrete']
    Type        = data['Type']

    w = np.where(
        (Type == 0) & (ICS_z0 > 0) & (Mvir >= MVIR_THRESHOLD_FORMATION) &
        (StellarMass >= STELLARMASS_THRESHOLD) & (n_satellites >= MIN_SATELLITES)
    )[0]

    if len(w) == 0:
        print('  No central galaxies with ICS > 0 found.')
        return

    Mvir_sel  = Mvir[w]
    ICS_d_sel = ICS_d_all[w]
    ICS_a_sel = ICS_a_all[w]

    ICS_total = ICS_d_sel + ICS_a_sel

    # Guard: centrals with zero computed total (e.g. model run before this feature)
    # fall back to treating total ICS as pure disruption
    zero_mask = ICS_total == 0
    ICS_d_sel[zero_mask] = ICS_z0[w][zero_mask]
    ICS_total = ICS_d_sel + ICS_a_sel

    f_accretion = np.where(ICS_total > 0, ICS_a_sel / ICS_total, 0.0)
    log_Mvir    = np.log10(Mvir_sel)
    log_ICS_tot = np.log10(ICS_total)

    # ------------------------------------------------------------------
    # Plot 1: M_ICS_total vs M_vir coloured by f_accretion
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(log_Mvir, log_ICS_tot,
                    c=f_accretion, cmap='plasma', vmin=0, vmax=1,
                    s=40, alpha=0.7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$f_{\mathrm{accretion}} = M_{\mathrm{ICS,acc}} / M_{\mathrm{ICS,total}}$',
                 fontsize=12)
    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'$\log_{10} m_{\mathrm{ICS,total}}\ [M_{\odot}]$', fontsize=14)
    ax.set_title(rf'$z \approx {z:.2f}$ — ICS channel (dodgerblue = disruption, red = accretion)',
                 fontsize=11)
    
    ax.set_xlim(np.log10(MVIR_THRESHOLD_FORMATION) - 0.1, 15)

    plt.tight_layout()
    outputFile1 = os.path.join(output_dir, 'ICS_mass_vs_mvir_by_channel' + OutputFormat)
    plt.savefig(outputFile1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile1}')

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    both_pos  = (ICS_d_sel > 0) & (ICS_a_sel > 0)
    n_total        = len(w)
    n_disrupt_only = int(np.sum(ICS_a_sel == 0))
    n_both         = int(np.sum(both_pos))
    n_acc_dom      = int(np.sum(f_accretion >= 0.5))

    print(f'  Snapshot : {snap}  (z = {z:.2f})')
    print(f'  Centrals with ICS > 0, Mvir >= 10^{np.log10(MVIR_THRESHOLD_FORMATION):.1f}: {n_total}')
    print(f'  Disruption only (satellite had no ICS): {n_disrupt_only}  ({100*n_disrupt_only/n_total:.1f}%)')
    print(f'  Both channels present                 : {n_both}  ({100*n_both/n_total:.1f}%)')
    print(f'  Accretion-dominated (f_acc >= 0.5)    : {n_acc_dom}  ({100*n_acc_dom/n_total:.1f}%)')
    print(f'  Median f_accretion                    : {np.median(f_accretion):.3f}')
    print(f'  Mean   f_accretion                    : {np.mean(f_accretion):.3f}')


def plot_ics_channel_vs_redshift(file_list, sim_params, output_dir, max_redshift=2.0, cache=None):
    """
    Plot ICS formation channel mass vs redshift.

    Shows how ICS_disrupt and ICS_accrete evolve with redshift using median lines with shading.

    Parameters:
    -----------
    file_list : list
        List of HDF5 file paths
    sim_params : dict
        Simulation parameters from read_simulation_params()
    output_dir : str
        Directory to save plots
    max_redshift : float
        Maximum redshift to include (default 2.0)
    cache : DataCache, optional
        Data cache for efficient data reuse.
    """
    print('\nRunning ICS Formation Channel vs Redshift Plot\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    available_snapshots = sim_params['available_snapshots']

    # Filter snapshots to those with z <= max_redshift
    snaps_to_plot = [s for s in available_snapshots
                     if s < len(redshifts) and redshifts[s] <= max_redshift]
    snaps_to_plot.sort(reverse=True)  # Start from highest redshift

    if not snaps_to_plot:
        print('  No snapshots available in redshift range.')
        return

    # Track statistics per snapshot for each channel
    snap_z = []
    # ICS_disrupt statistics
    snap_median_disc = []
    snap_p16_disc = []
    snap_p84_disc = []
    # ICS_accrete statistics
    snap_median_acc = []
    snap_p16_acc = []
    snap_p84_acc = []
    snap_n_galaxies = []

    for snap in snaps_to_plot:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        # Read data
        fields = ['Mvir', 'IntraClusterStars', 'StellarMass', 'ICS_disrupt', 'ICS_accrete', 'Type',
                  'GalaxyIndex', 'CentralGalaxyIndex']
        try:
            data = read_hdf_multi_params(file_list, Snapshot, fields)
            Mvir = data.get('Mvir', np.array([])) * 1.0e10 / Hubble_h
            ICS = data.get('IntraClusterStars', np.array([])) * 1.0e10 / Hubble_h
            StellarMass = data.get('StellarMass', np.array([])) * 1.0e10 / Hubble_h
            Type = data.get('Type', np.array([]))
            ICS_disrupt = data.get('ICS_disrupt', np.array([])) * 1.0e10 / Hubble_h
            ICS_accrete = data.get('ICS_accrete', np.array([])) * 1.0e10 / Hubble_h
            GalaxyIndex = data.get('GalaxyIndex', np.array([]))
            CentralGalaxyIndex = data.get('CentralGalaxyIndex', np.array([]))
        except Exception as e:
            print(f'  Error reading snapshot {snap}: {e}')
            continue

        if ICS_disrupt.size == 0 or ICS_accrete.size == 0:
            continue

        # Count satellites per central
        sorted_idx = np.argsort(GalaxyIndex)
        sorted_gids = GalaxyIndex[sorted_idx]
        sat_central_gids = CentralGalaxyIndex[Type != 0]
        ins_pos = np.searchsorted(sorted_gids, sat_central_gids)
        ins_pos = np.clip(ins_pos, 0, len(sorted_gids) - 1)
        c_idx = np.where(sorted_gids[ins_pos] == sat_central_gids, sorted_idx[ins_pos], -1)
        n_satellites = np.zeros(len(Type), dtype=int)
        np.add.at(n_satellites, c_idx[c_idx >= 0], 1)

        # Select centrals with ICS > 0 and above thresholds
        w = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_FORMATION) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (ICS > 0) & (n_satellites >= MIN_SATELLITES))[0]

        if len(w) == 0:
            continue

        ICS_d = ICS_disrupt[w]
        ICS_a = ICS_accrete[w]

        # Only include galaxies with non-zero values for statistics
        valid_disc = ICS_d > 0
        valid_acc = ICS_a > 0

        if not np.any(valid_disc) or not np.any(valid_acc):
            continue

        # Store redshift
        snap_z.append(z)
        snap_n_galaxies.append(len(w))

        # Compute statistics for disruption channel (log scale)
        log_disc = np.log10(ICS_d[valid_disc])
        snap_median_disc.append(np.median(log_disc))
        snap_p16_disc.append(np.percentile(log_disc, 16))
        snap_p84_disc.append(np.percentile(log_disc, 84))

        # Compute statistics for accretion channel (log scale)
        log_acc = np.log10(ICS_a[valid_acc])
        snap_median_acc.append(np.median(log_acc))
        snap_p16_acc.append(np.percentile(log_acc, 16))
        snap_p84_acc.append(np.percentile(log_acc, 84))

    if not snap_z:
        print('  No data to plot.')
        return

    # Convert to arrays and sort by redshift
    snap_z = np.array(snap_z)
    sort_idx = np.argsort(snap_z)
    snap_z = snap_z[sort_idx]

    snap_median_disc = np.array(snap_median_disc)[sort_idx]
    snap_p16_disc = np.array(snap_p16_disc)[sort_idx]
    snap_p84_disc = np.array(snap_p84_disc)[sort_idx]

    snap_median_acc = np.array(snap_median_acc)[sort_idx]
    snap_p16_acc = np.array(snap_p16_acc)[sort_idx]
    snap_p84_acc = np.array(snap_p84_acc)[sort_idx]

    # ================================================================
    # Plot: ICS mass vs redshift with median lines and shading
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Disruption channel (dodgerblue)
    ax.fill_between(snap_z, snap_p16_disc, snap_p84_disc,
                    color='gray', alpha=0.3, label=None)
    ax.plot(snap_z, snap_median_disc, color='k', lw=2.5, linestyle='--',
            label=r'Disruption')

    # Accretion channel (red)
    ax.fill_between(snap_z, snap_p16_acc, snap_p84_acc,
                    color='gray', alpha=0.3, label=None)
    ax.plot(snap_z, snap_median_acc, color='k', lw=2.5, linestyle='-',
            label=r'Accretion')

    ax.set_xlabel(r'Redshift $z$', fontsize=14)
    ax.set_ylabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14)
    ax.set_xlim(0, max_redshift)
    ax.legend(loc='upper right', fontsize=11, frameon=False)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'ICS_channel_vs_redshift' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outputFile}')

    # ================================================================
    # Print statistics
    # ================================================================
    print(f'  Snapshots: {len(snap_z)} (z = 0 to {max_redshift})')
    z0_idx = np.argmin(snap_z)
    print(f'  At z=0: median log(ICS_disrupt) = {snap_median_disc[z0_idx]:.2f}, median log(ICS_accrete) = {snap_median_acc[z0_idx]:.2f}')
    if snap_z.max() > 0:
        zmax_idx = np.argmax(snap_z)
        print(f'  At z={snap_z.max():.1f}: median log(ICS_disrupt) = {snap_median_disc[zmax_idx]:.2f}, median log(ICS_accrete) = {snap_median_acc[zmax_idx]:.2f}')


def plot_bcg_metallicity_vs_stellar_mass(file_list, sim_params, output_dir, cache=None):
    """
    Plot BCG metallicity (Z/Z_sun) as a function of stellar mass for central galaxies
    at multiple redshifts, tracking the same BCGs (via GalaxyIndex) across time.

    Metallicity is calculated as (MetalsStellarMass / StellarMass) / 0.02
    where 0.02 is the solar metallicity reference.
    """
    print('\nRunning BCG Metallicity vs Stellar Mass Plot (Progenitor Tracking)\n')

    Hubble_h = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    last_snap = sim_params['last_snapshot']

    # Target redshifts to plot
    target_z = [0, 0.2, 0.5, 1, 1.5, 2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Find closest snapshots to target redshifts
    snap_z_pairs = []
    for tz in target_z:
        snap = np.argmin(np.abs(np.array(redshifts) - tz))
        actual_z = redshifts[snap]
        snap_z_pairs.append((snap, actual_z))

    # Step 1: Get z=0 BCGs and their GalaxyIndex (for tracking as centrals)
    Snapshot_z0 = 'Snap_' + str(last_snap)
    if cache is not None:
        data_z0 = cache.get(Snapshot_z0, ['Mvir', 'StellarMass', 'MetalsStellarMass', 'Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    else:
        data_z0 = {
            'StellarMass': read_hdf(file_list, Snapshot_z0, 'StellarMass') * 1.0e10 / Hubble_h,
            'Mvir': read_hdf(file_list, Snapshot_z0, 'Mvir') * 1.0e10 / Hubble_h,
            'MetalsStellarMass': read_hdf(file_list, Snapshot_z0, 'MetalsStellarMass') * 1.0e10 / Hubble_h,
            'Type': read_hdf(file_list, Snapshot_z0, 'Type'),
            'GalaxyIndex': read_hdf(file_list, Snapshot_z0, 'GalaxyIndex'),
            'CentralGalaxyIndex': read_hdf(file_list, Snapshot_z0, 'CentralGalaxyIndex'),
        }

    # Select z=0 BCGs: centrals (Type==0) where GalaxyIndex == CentralGalaxyIndex
    w_z0 = np.where((data_z0['Type'] == 0) &
                    (data_z0['GalaxyIndex'] == data_z0['CentralGalaxyIndex']) &
                    (data_z0['StellarMass'] > STELLARMASS_THRESHOLD) &
                    (data_z0['Mvir'] >= MVIR_THRESHOLD_HALOMASS) &
                    (data_z0['MetalsStellarMass'] > 0))[0]

    if len(w_z0) == 0:
        print('  No z=0 BCGs found matching criteria.')
        return

    # Get the GalaxyIndex of z=0 BCGs for tracking
    z0_galaxy_indices = data_z0['GalaxyIndex'][w_z0]
    n_bcgs = len(z0_galaxy_indices)
    print(f'  Tracking {n_bcgs} z=0 BCGs back in time')

    # Create set for fast lookup
    z0_bcg_set = set(z0_galaxy_indices)

    # Create plot
    plt.figure(figsize=(10, 7))

    # Step 2: For each target redshift, find progenitors by matching CentralGalaxyIndex
    for i, (snap, z) in enumerate(snap_z_pairs):
        Snapshot = 'Snap_' + str(snap)

        # Read data for this snapshot
        if cache is not None:
            data = cache.get(Snapshot, ['StellarMass', 'MetalsStellarMass', 'Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
        else:
            data = {
                'StellarMass': read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h,
                'MetalsStellarMass': read_hdf(file_list, Snapshot, 'MetalsStellarMass') * 1.0e10 / Hubble_h,
                'Type': read_hdf(file_list, Snapshot, 'Type'),
                'GalaxyIndex': read_hdf(file_list, Snapshot, 'GalaxyIndex'),
                'CentralGalaxyIndex': read_hdf(file_list, Snapshot, 'CentralGalaxyIndex'),
            }

        StellarMass = data['StellarMass']
        MetalsStellarMass = data['MetalsStellarMass']
        Type = data['Type']
        GalaxyIndex = data['GalaxyIndex']
        CentralGalaxyIndex = data['CentralGalaxyIndex']

        # Find BCGs: centrals where GalaxyIndex == CentralGalaxyIndex and in our tracked set
        bcg_mask = ((Type == 0) &
                    (GalaxyIndex == CentralGalaxyIndex) &
                    (MetalsStellarMass > 0) &
                    (StellarMass > 0))

        bcg_gal_idx = GalaxyIndex[bcg_mask]
        bcg_stellar = StellarMass[bcg_mask]
        bcg_metals = MetalsStellarMass[bcg_mask]

        # Match to z=0 BCG set
        is_tracked = np.array([gid in z0_bcg_set for gid in bcg_gal_idx])

        if not np.any(is_tracked):
            print(f'  z = {z:.2f}: No progenitors found')
            continue

        # Get data for matched progenitors
        prog_stellar = bcg_stellar[is_tracked]
        prog_metals = bcg_metals[is_tracked]

        # Calculate metallicity as Z/Z_sun where Z_sun = 0.02
        metallicity = (prog_metals / prog_stellar) / 0.02

        plt.scatter(np.log10(prog_stellar), metallicity,
                   s=37, alpha=0.7, color=colors[i], edgecolor='none',
                   rasterized=True, marker='o', label=f'z = {z:.1f}')

        print(f'  z = {z:.2f}: {np.sum(is_tracked)}/{n_bcgs} progenitors found, '
              f'median Z/Z_sun = {np.median(metallicity):.3f}')

    plt.xlabel(r'$\log_{10} M_{\mathrm{BCG}}\ [M_{\odot}]$', fontsize=14)
    plt.ylabel(r'$Z / Z_{\odot}$', fontsize=14)
    plt.xlim(10, 12.5)
    plt.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = os.path.join(output_dir, 'BCG_metallicity_vs_stellar_mass' + OutputFormat)
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {outputFile}')


# ==================================================================
# MAIN
# ==================================================================

if __name__ == '__main__':

    print('Running BCG and ICS Analysis Plots\n')
    print(f'  Mvir threshold: {MVIR_THRESHOLD_REDSHIFT:.2e} Msun (log10 = {np.log10(MVIR_THRESHOLD_REDSHIFT):.1f})')
    print(f'  StellarMass threshold: {STELLARMASS_THRESHOLD:.2e} Msun (log10 = {np.log10(STELLARMASS_THRESHOLD):.1f})')

    # Parse command-line arguments
    args = parse_arguments()

    # Determine paths and find files
    script_dir = get_script_dir()

    # Use glob to find all files matching the pattern
    file_list = glob.glob(args.input_pattern)
    file_list.sort()  # Ensure consistent ordering

    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f'\nFound {len(file_list)} model files.')

    first_file = os.path.abspath(file_list[0])
    input_dir = os.path.dirname(first_file)

    # Read simulation parameters from the first HDF5 header
    print(f'Reading simulation parameters from {first_file}')
    sim_params = read_simulation_params(first_file)

    # Calculate the total volume fraction across ALL files
    total_volume_fraction = 0.0
    for f in file_list:
        p = read_simulation_params(f)
        total_volume_fraction += p['VolumeFraction']

    sim_params['VolumeFraction'] = total_volume_fraction

    Hubble_h = sim_params['Hubble_h']
    BoxSize = sim_params['BoxSize']
    VolumeFraction = sim_params['VolumeFraction']
    redshifts = sim_params['redshifts']

    print(f'  Hubble_h = {Hubble_h}')
    print(f'  BoxSize = {BoxSize} h^-1 Mpc')
    print(f'  Total VolumeFraction = {VolumeFraction}')
    print(f'  Available snapshots: {sim_params["first_snapshot"]} to {sim_params["last_snapshot"]}')

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(input_dir, 'plots')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'  Output directory: {output_dir}')

    # Create data cache for efficient data reuse between plots
    cache = DataCache(file_list, Hubble_h, sim_params)

    # ==================================================================
    # Run all plots
    # ==================================================================

    plot_ics_mass_function(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_ratio_distribution(file_list, sim_params, output_dir, cache=cache)
    plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir, cache=cache)
    plot_ics_stellar_fraction_vs_redshift(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_fraction_vs_redshift(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_ratio_vs_redshift(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_ratio_vs_halomass(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_fraction_vs_halomass(file_list, sim_params, output_dir, cache=cache)
    plot_ics_bcg_fraction_vs_formation_z(file_list, sim_params, output_dir, cache=cache)
    plot_ICSmass_vs_halomass(file_list, sim_params, output_dir, cache=cache)
    plot_ics_vs_bcg_mass(file_list, sim_params, output_dir, cache=cache)
    plot_ICS_mass_vs_mvir_by_channel(file_list, sim_params, output_dir, cache=cache)
    plot_ics_channel_vs_redshift(file_list, sim_params, output_dir, cache=cache)
    ## plot_ics_formation_channels(file_list, sim_params, output_dir, cache=cache)
    plot_ics_channel_breakdown(file_list, sim_params, output_dir, cache=cache)
    plot_ics_channel_fraction_grid(file_list, sim_params, output_dir, cache=cache)
    plot_icl_contribution_by_infall_mass(file_list, sim_params, output_dir, cache=cache)
    plot_bcg_metallicity_vs_stellar_mass(file_list, sim_params, output_dir, cache=cache)


    print('\nAll plots completed!')


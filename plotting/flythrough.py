#!/usr/bin/env python
"""
SAGE26 Simulation Flythrough Generator

Creates animated flythroughs of the simulation box using PyVista.
Supports 4 animation modes:
    1. Camera orbit - Rotate around the box at a single snapshot
    2. Fly through box - Camera travels through the box along a path
    3. Time evolution - Watch galaxies form from high-z to z=0
    4. Combined - Time evolution with camera motion

Usage:
    python flythrough.py --mode orbit --color-by density
    python flythrough.py --mode flythrough --color-by mass
    python flythrough.py --mode evolution
    python flythrough.py --mode combined
    python flythrough.py --mode orbit --halo-mass-cmap viridis  # custom halo colormap

Author: Generated for SAGE26 SAM visualization
"""

import h5py as h5
import numpy as np
import pyvista as pv
import argparse
import os
from scipy.stats import gaussian_kde

# MPI support (optional - falls back to serial if not available)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_ENABLED = size > 1
except ImportError:
    comm = None
    rank = 0
    size = 1
    MPI_ENABLED = False


def mpi_print(*args, **kwargs):
    """Print only from rank 0."""
    if rank == 0:
        print(*args, **kwargs)


def mpi_barrier():
    """Synchronize all MPI ranks."""
    if comm is not None:
        comm.Barrier()

# ========================== CONFIGURATION ==========================

# File paths (relative to SAGE26 root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'output', 'millennium')
DATA_FILE = os.path.join(DATA_DIR, 'model_0.hdf5')
OUTPUT_DIR = os.path.join(DATA_DIR, 'movies')
TREE_DIR = os.path.join(ROOT_DIR, 'input', 'millennium', 'trees')

# Simulation parameters
HUBBLE_H = 0.73
BOX_SIZE = 62.5  # Mpc/h

# Animation parameters
FPS = 30
ORBIT_DURATION = 5      # seconds for one full orbit
NUM_ORBITS = 2           # number of orbits (total duration = ORBIT_DURATION * NUM_ORBITS)
FLYTHROUGH_DURATION = 60  # seconds for flythrough (longer = smoother)
EVOLUTION_DURATION = 10   # seconds for time evolution
COMBINED_DURATION = 10    # seconds for combined animation

# Output format: 'mp4', 'mov', 'gif', or 'frames' (individual PNG files)
OUTPUT_FORMAT = 'frames'

# Galaxy selection
MIN_STELLAR_MASS = 1.0e8  # Minimum stellar mass in Msun (after unit conversion)
MAX_GALAXIES = 50000      # Maximum galaxies to render (for performance)

# Visual settings
BACKGROUND_COLOR = 'black'
BOX_COLOR = 'white'
BOX_OPACITY = 0.3
SHOW_BOX = False  # Set to True to show simulation box wireframe
GALAXY_OPACITY = 0.17  # Opacity for galaxy points (0.0 - 1.0)

# Particle size bins (can add more bins for finer gradation)
HALO_SIZE_BINS = [25.0, 30.0, 35.0, 40.0, 60.0]  # Size bins for halos
GALAXY_SIZE_SCALE = 0.17  # Galaxy sizes = halo sizes * this factor

# Coloring Modes
COLOR_MODE = 'mass'     # Default mode: 'mass', 'density', 'sfr', or 'type'

# Galaxy colormaps
MASS_COLORMAP = 'plasma'    # Colormap for Mass mode
DENSITY_COLORMAP = 'magma'  # Colormap for Density mode
SFR_COLORMAP = 'coolwarm_r'    # Colormap for sSFR mode (specific star formation rate)
CENTRAL_COLORMAP = 'Blues_r'   # Colormap for central galaxies (Type mode, colored by mass)
SATELLITE_COLORMAP = 'Reds_r'  # Colormap for satellite galaxies (Type mode, colored by mass)

# Halo colormaps (can be different from galaxy colormaps)
HALO_MASS_COLORMAP = 'Blues'      # Colormap for halos in Mass mode
HALO_DENSITY_COLORMAP = 'magma'    # Colormap for halos in Density mode
HALO_SFR_COLORMAP = 'coolwarm_r'       # Colormap for halos in sSFR mode (halos use mass coloring)
HALO_TYPE_COLORMAP = 'Blues'      # Colormap for halos in Type mode (halos use mass coloring)

# Colormap ranges (set to None for auto-scaling based on data)
# Values are in log10 units where applicable
STELLAR_MASS_RANGE = [8.0, 12.0]   # log10(Msun) range for mass coloring and size scaling
SSFR_RANGE = [-14.0, -8.0]         # log10(yr^-1) range for sSFR coloring
DENSITY_RANGE = None               # Set to [min, max] to fix density range, or None for auto
HALO_MASS_RANGE = [10.0, 15.0]     # log10(Msun) range for halo size scaling

# Halo visualization settings
SHOW_HALOS = True  # Set to True to show dark matter halos
HALO_MIN_MASS = 1.0e10  # Minimum halo mass to show (Msun)
HALO_OPACITY = 0.025  # Transparency of halo points

# Full redshift list for all 64 snapshots (Snap_0 to Snap_63)
SNAPSHOT_REDSHIFTS = [
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941,
    11.897, 10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724,
    5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 2.831, 2.619,
    2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173,
    1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457,
    0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 0.116, 0.089,
    0.064, 0.041, 0.020, 0.000
]

# ==================================================================


def read_hdf(filename, snap_num, param):
    """Read a parameter from the HDF5 file for a given snapshot."""
    with h5.File(filename, 'r') as f:
        return np.array(f[snap_num][param])


# Define the halo data structure (matches C struct in SAGE)
HALO_DTYPE = np.dtype([
    ('Descendant', np.int32),
    ('FirstProgenitor', np.int32),
    ('NextProgenitor', np.int32),
    ('FirstHaloInFOFgroup', np.int32),
    ('NextHaloInFOFgroup', np.int32),
    ('Len', np.int32),
    ('M_Mean200', np.float32),
    ('Mvir', np.float32),
    ('M_TopHat', np.float32),
    ('Pos', np.float32, (3,)),
    ('Vel', np.float32, (3,)),
    ('VelDisp', np.float32),
    ('Vmax', np.float32),
    ('Spin', np.float32, (3,)),
    ('MostBoundID', np.int64),
    ('SnapNum', np.int32),
    ('FileNr', np.int32),
    ('SubhaloIndex', np.int32),
    ('SubHalfMass', np.float32),
])


def load_halo_data(tree_dir, snapshot_num, mass_cut=HALO_MIN_MASS, max_halos=50000):
    """Load halo positions from binary tree files for a given snapshot."""
    mpi_print(f"  Loading halos for snapshot {snapshot_num}...")

    all_positions = []
    all_masses = []

    # Read all tree files
    for file_num in range(8):
        tree_file = os.path.join(tree_dir, f'trees_063.{file_num}')
        if not os.path.exists(tree_file):
            continue

        with open(tree_file, 'rb') as f:
            nforests = np.fromfile(f, dtype=np.int32, count=1)[0]
            nhalos_total = np.fromfile(f, dtype=np.int32, count=1)[0]

            if nhalos_total == 0:
                continue

            nhalos_per_forest = np.fromfile(f, dtype=np.int32, count=nforests)
            halos = np.fromfile(f, dtype=HALO_DTYPE, count=nhalos_total)

            mask = halos['SnapNum'] == snapshot_num
            halos_snap = halos[mask]

            if len(halos_snap) > 0:
                positions = halos_snap['Pos']
                masses = halos_snap['Mvir'] * 1.0e10 / HUBBLE_H

                mass_mask = masses > mass_cut
                all_positions.append(positions[mass_mask])
                all_masses.append(masses[mass_mask])

    if len(all_positions) == 0:
        mpi_print(f"    No halos found for snapshot {snapshot_num}")
        return np.array([]).reshape(0, 3), np.array([])

    positions = np.vstack(all_positions)
    masses = np.concatenate(all_masses)

    if len(positions) > max_halos:
        np.random.seed(42)
        idx = np.random.choice(len(positions), max_halos, replace=False)
        positions = positions[idx]
        masses = masses[idx]

    mpi_print(f"    Selected {len(positions)} halos")
    return positions, masses


def get_halo_sizes(masses):
    """Scale halo point sizes by mass using fixed HALO_MASS_RANGE to prevent flickering."""
    if len(masses) == 0: return np.array([])
    min_size, max_size = HALO_SIZE_BINS[0], HALO_SIZE_BINS[-1]
    log_mass = np.log10(masses + 1)
    vmin, vmax = HALO_MASS_RANGE
    normalized = (log_mass - vmin) / (vmax - vmin + 1e-10)
    return min_size + np.clip(normalized, 0, 1) * (max_size - min_size)


def add_halos_to_plotter(plotter, positions, masses, colors=None, opacity_scale=1.0):
    """
    Add halo point cloud to the plotter.
    If 'colors' is None, defaults to mass-based coloring.
    opacity_scale: multiplier for opacity (used during crossfade transitions)
    """
    if len(positions) == 0:
        return

    sizes = get_halo_sizes(masses)

    # Determine coloring scheme based on current COLOR_MODE
    if colors is None:
        # Normalize halo masses to 0-1 using fixed range (prevents flickering)
        log_mass = np.log10(masses + 1)
        vmin, vmax = HALO_MASS_RANGE
        colors = np.clip((log_mass - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # Select halo colormap based on current mode
    if COLOR_MODE == 'density':
        cmap = HALO_DENSITY_COLORMAP
    elif COLOR_MODE == 'sfr':
        cmap = HALO_SFR_COLORMAP
    elif COLOR_MODE == 'type':
        cmap = HALO_TYPE_COLORMAP
    else:  # mass
        cmap = HALO_MASS_COLORMAP

    # Use configurable size bins
    for i, max_s in enumerate(HALO_SIZE_BINS):
        min_s = HALO_SIZE_BINS[i-1] if i > 0 else 0
        mask = (sizes >= min_s) & (sizes < max_s) if i < len(HALO_SIZE_BINS)-1 else (sizes >= min_s)

        if np.sum(mask) > 0:
            cloud = pv.PolyData(positions[mask])
            # Store values in mesh
            cloud['values'] = colors[mask]

            plotter.add_mesh(
                cloud,
                scalars='values',
                cmap=cmap,
                clim=[0, 1],
                point_size=max_s,
                render_points_as_spheres=True,
                opacity=HALO_OPACITY * opacity_scale,
                show_scalar_bar=False
            )


def get_snapshot_redshift(snap_num):
    snap_idx = int(snap_num.replace('Snap_', ''))
    if 0 <= snap_idx < len(SNAPSHOT_REDSHIFTS):
        return SNAPSHOT_REDSHIFTS[snap_idx]
    return 0.0


def load_galaxy_data(filename, snapshot, mass_cut=MIN_STELLAR_MASS, max_gals=MAX_GALAXIES):
    """Load galaxy positions and properties including SFR and Type."""
    mpi_print(f"  Loading {snapshot}...")
    try:
        posx = read_hdf(filename, snapshot, 'Posx')
        posy = read_hdf(filename, snapshot, 'Posy')
        posz = read_hdf(filename, snapshot, 'Posz')
        stellar_mass = read_hdf(filename, snapshot, 'StellarMass') * 1.0e10 / HUBBLE_H
        mvir = read_hdf(filename, snapshot, 'Mvir')
        sfr_disk = read_hdf(filename, snapshot, 'SfrDisk')
        sfr_bulge = read_hdf(filename, snapshot, 'SfrBulge')
        sfr = sfr_disk + sfr_bulge  # Total star formation rate
        ssfr = sfr / (stellar_mass + 1e-10)  # Specific SFR (SFR / stellar mass)
        gal_type = read_hdf(filename, snapshot, 'Type')  # 0=central, 1+=satellite
    except KeyError as e:
        mpi_print(f"    Warning: Missing field {e}")
        return np.array([]).reshape(0,3), np.array([]), np.array([]), np.array([])

    mask = (stellar_mass > mass_cut) & (mvir > 0)
    indices = np.where(mask)[0]

    if len(indices) > max_gals:
        np.random.seed(42)
        indices = np.random.choice(indices, max_gals, replace=False)

    positions = np.column_stack([posx[indices], posy[indices], posz[indices]])
    mpi_print(f"    Selected {len(indices)} galaxies")
    return positions, stellar_mass[indices], ssfr[indices], gal_type[indices]


def create_box_mesh():
    box = pv.Box(bounds=(0, BOX_SIZE, 0, BOX_SIZE, 0, BOX_SIZE))
    edges = box.extract_all_edges()
    return edges


def compute_density_colors(positions):
    """Compute KDE-based density coloring for galaxies/halos using DENSITY_RANGE."""
    if len(positions) < 10:
        return np.ones(len(positions))

    mpi_print("    Computing density estimates...")
    # Subsample for KDE computation to keep it fast
    if len(positions) > 5000:
        np.random.seed(42)  # Fixed seed for consistent results across frames
        sample_idx = np.random.choice(len(positions), 5000, replace=False)
        kde_data = positions[sample_idx].T
    else:
        kde_data = positions.T

    try:
        kde = gaussian_kde(kde_data)
        # Evaluate density on all points
        density = kde(positions.T)

        # Log scaling for better visual dynamic range
        density = np.log10(density + 1e-10)

        # Use configured range or auto-scale
        if DENSITY_RANGE is not None:
            d_min, d_max = DENSITY_RANGE
        else:
            d_min, d_max = density.min(), density.max()

        if d_max > d_min:
            density = (density - d_min) / (d_max - d_min)
        else:
            density = np.zeros_like(density)

        return np.clip(density, 0, 1)
    except Exception as e:
        mpi_print(f"    Density computation failed: {e}")
        return np.zeros(len(positions))


def get_mass_colors(stellar_mass):
    """Normalize stellar mass to 0-1 for coloring using STELLAR_MASS_RANGE."""
    log_mass = np.log10(stellar_mass + 1)
    if STELLAR_MASS_RANGE is not None:
        vmin, vmax = STELLAR_MASS_RANGE
    else:
        vmin, vmax = log_mass.min(), log_mass.max()
    normalized = (log_mass - vmin) / (vmax - vmin + 1e-10)
    return np.clip(normalized, 0, 1)


def get_ssfr_colors(ssfr):
    """Normalize sSFR to 0-1 for coloring using SSFR_RANGE."""
    # Use log scale, handling zero/negative sSFR
    ssfr_safe = np.maximum(ssfr, 1e-14)
    log_ssfr = np.log10(ssfr_safe)
    if SSFR_RANGE is not None:
        vmin, vmax = SSFR_RANGE
    else:
        vmin, vmax = log_ssfr.min(), log_ssfr.max()
    normalized = (log_ssfr - vmin) / (vmax - vmin + 1e-10)
    return np.clip(normalized, 0, 1)


def get_mass_sizes(stellar_mass):
    """Scale galaxy point sizes by mass using fixed STELLAR_MASS_RANGE to prevent flickering."""
    min_size = HALO_SIZE_BINS[0] * GALAXY_SIZE_SCALE
    max_size = HALO_SIZE_BINS[-1] * GALAXY_SIZE_SCALE
    log_mass = np.log10(stellar_mass + 1)
    vmin, vmax = STELLAR_MASS_RANGE
    normalized = (log_mass - vmin) / (vmax - vmin + 1e-10)
    return min_size + np.clip(normalized, 0, 1) * (max_size - min_size)


def setup_plotter(off_screen=True):
    plotter = pv.Plotter(off_screen=off_screen, window_size=[1920, 1080])
    plotter.set_background(BACKGROUND_COLOR)
    return plotter


def add_galaxies_to_plotter(plotter, positions, colors, sizes=None, opacity_scale=1.0,
                            gal_type=None, mass_colors=None):
    """
    Add galaxies to plotter.

    For 'type' mode: pass gal_type array and mass_colors for coloring by mass within each type.
    """
    if len(positions) == 0:
        return

    # Base opacity from config, scaled by transition factor
    opacity = GALAXY_OPACITY * opacity_scale
    galaxy_size_bins = [s * GALAXY_SIZE_SCALE for s in HALO_SIZE_BINS]

    # Type mode: render centrals and satellites separately with different colormaps
    if COLOR_MODE == 'type' and gal_type is not None and mass_colors is not None:
        # Centrals (type == 0)
        central_mask = (gal_type == 0)
        if np.sum(central_mask) > 0:
            _render_galaxy_subset(plotter, positions[central_mask], mass_colors[central_mask],
                                  sizes[central_mask] if sizes is not None else None,
                                  CENTRAL_COLORMAP, opacity, galaxy_size_bins)

        # Satellites (type > 0)
        sat_mask = (gal_type > 0)
        if np.sum(sat_mask) > 0:
            _render_galaxy_subset(plotter, positions[sat_mask], mass_colors[sat_mask],
                                  sizes[sat_mask] if sizes is not None else None,
                                  SATELLITE_COLORMAP, opacity, galaxy_size_bins)
    else:
        # Standard modes: mass, density, sfr
        if COLOR_MODE == 'density':
            cmap = DENSITY_COLORMAP
        elif COLOR_MODE == 'sfr':
            cmap = SFR_COLORMAP
        else:  # mass
            cmap = MASS_COLORMAP

        _render_galaxy_subset(plotter, positions, colors, sizes, cmap, opacity, galaxy_size_bins)


def _render_galaxy_subset(plotter, positions, colors, sizes, cmap, opacity, size_bins):
    """Helper to render a subset of galaxies with given colormap."""
    if len(positions) == 0:
        return

    if sizes is not None:
        for i, max_s in enumerate(size_bins):
            min_s = size_bins[i-1] if i > 0 else 0
            mask = (sizes >= min_s) & (sizes < max_s) if i < len(size_bins)-1 else (sizes >= min_s)
            if np.sum(mask) > 0:
                bin_cloud = pv.PolyData(positions[mask])
                bin_cloud['colors'] = colors[mask]
                plotter.add_mesh(
                    bin_cloud,
                    scalars='colors',
                    cmap=cmap,
                    clim=[0, 1],
                    point_size=max_s,
                    render_points_as_spheres=True,
                    opacity=opacity,
                    show_scalar_bar=False
                )
    else:
        cloud = pv.PolyData(positions)
        cloud['colors'] = colors
        plotter.add_mesh(
            cloud,
            scalars='colors',
            cmap=cmap,
            clim=[0, 1],
            point_size=POINT_SIZE,
            render_points_as_spheres=True,
            opacity=opacity,
            show_scalar_bar=False
        )


def add_box_to_plotter(plotter):
    box = create_box_mesh()
    plotter.add_mesh(box, color=BOX_COLOR, line_width=1, opacity=BOX_OPACITY)


def add_text_annotation(plotter, text, position='upper_left', font_size=14):
    plotter.add_text(text, position=position, font_size=font_size, color='white')


def check_existing_frames(frames_dir, expected_count):
    if not os.path.exists(frames_dir): return False
    existing = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')]
    if len(existing) >= expected_count:
        mpi_print(f"  Found {len(existing)} existing frames in {frames_dir}/")
        return True
    return False


class FrameWriter:
    def __init__(self, plotter, output_path, fps=FPS, output_format=None, expected_frames=None):
        self.plotter = plotter
        self.output_path = output_path
        self.fps = fps
        self.format = output_format if output_format is not None else OUTPUT_FORMAT
        self.frame_count = 0
        self.frames_dir = None
        self.writer = None
        self.skip_rendering = False

        if self.format in ('frames', 'mov'):
            base = os.path.splitext(output_path)[0]
            self.frames_dir = base + '_frames'
            if expected_frames and check_existing_frames(self.frames_dir, expected_frames):
                self.skip_rendering = True
                mpi_print(f"  Skipping render - using existing frames")
            else:
                os.makedirs(self.frames_dir, exist_ok=True)
                mpi_print(f"  Saving frames to: {self.frames_dir}/")
        elif self.format == 'gif':
            import imageio
            self.writer = imageio.get_writer(output_path.replace('.mp4', '.gif'), mode='I', duration=1.0/fps)
        elif self.format == 'mp4':
            try:
                self.plotter.open_movie(output_path, framerate=fps, quality=8)
            except Exception as e:
                mpi_print(f"  Warning: MP4 creation failed ({e}), falling back to frames")
                self.format = 'frames'
                base = os.path.splitext(output_path)[0]
                self.frames_dir = base + '_frames'
                os.makedirs(self.frames_dir, exist_ok=True)

    def write_frame(self, frame_idx=None):
        """Write a frame. frame_idx specifies the frame number for MPI parallelism."""
        if self.skip_rendering:
            self.frame_count += 1
            return

        # Use provided frame_idx for naming (MPI), or fall back to internal counter
        idx = frame_idx if frame_idx is not None else self.frame_count

        if self.format in ('frames', 'mov'):
            frame_path = os.path.join(self.frames_dir, f'frame_{idx:05d}.png')
            self.plotter.screenshot(frame_path)
        elif self.format == 'gif':
            self.plotter.render()
            img = self.plotter.screenshot(return_img=True)
            self.writer.append_data(img)
        else:
            self.plotter.write_frame()
        self.frame_count += 1

    def close(self):
        import subprocess, shutil
        if self.format == 'gif' and self.writer: self.writer.close()
        elif self.format == 'mp4':
            try: self.plotter.close()
            except: pass

        # Only rank 0 runs ffmpeg conversion
        if rank != 0:
            return

        ffmpeg_available = shutil.which('ffmpeg') is not None
        if self.format in ('frames', 'mov') and ffmpeg_available:
            mpi_print(f"\n  Converting frames with ffmpeg...")
            ext = '.mov' if self.format == 'mov' else '.mp4'
            out_path = self.output_path.replace('.mp4', ext) if self.format == 'mov' else self.output_path
            cmd = ['ffmpeg', '-y', '-framerate', str(self.fps),
                   '-i', f'{self.frames_dir}/frame_%05d.png',
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path]
            subprocess.run(cmd, capture_output=True)
            mpi_print(f"  Created: {out_path}")


# ========================== ANIMATION MODES ==========================

def get_scene_colors(positions, stellar_mass, sfr=None, gal_type=None, halo_positions=None):
    """
    Helper to generate galaxy and halo colors based on current mode.

    Returns: (gal_colors, halo_colors, mass_colors, gal_type)
    - mass_colors is used for 'type' mode to color by mass within each type
    - gal_type is passed through for 'type' mode
    """
    halo_colors = None
    mass_colors = get_mass_colors(stellar_mass)  # Always compute for type mode

    if COLOR_MODE == 'density':
        gal_colors = compute_density_colors(positions)
        if halo_positions is not None and len(halo_positions) > 0:
            halo_colors = compute_density_colors(halo_positions)
    elif COLOR_MODE == 'sfr' and sfr is not None:
        gal_colors = get_ssfr_colors(sfr)  # sfr parameter is actually sSFR
        # Halos use mass colors in SFR mode
        halo_colors = None
    elif COLOR_MODE == 'type':
        # For type mode, gal_colors is mass_colors (used within each type)
        gal_colors = mass_colors
        halo_colors = None
    else:  # mass
        gal_colors = mass_colors
        halo_colors = None

    return gal_colors, halo_colors, mass_colors, gal_type


def create_orbit_animation(output_file, snapshot='Snap_63'):
    mpi_print(f"\nCreating orbit animation ({COLOR_MODE}) for {snapshot}...")
    mpi_print(f"  {NUM_ORBITS} orbit(s) x {ORBIT_DURATION}s = {NUM_ORBITS * ORBIT_DURATION}s total")
    if MPI_ENABLED:
        mpi_print(f"  Running with {size} MPI ranks")
    n_frames = int(ORBIT_DURATION * NUM_ORBITS * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    positions, stellar_mass, sfr, gal_type = load_galaxy_data(DATA_FILE, snapshot)
    sizes = get_mass_sizes(stellar_mass)
    redshift = get_snapshot_redshift(snapshot)

    halo_positions, halo_masses = None, None
    if SHOW_HALOS:
        snap_num = int(snapshot.replace('Snap_', ''))
        halo_positions, halo_masses = load_halo_data(TREE_DIR, snap_num)

    # Get colors based on current mode
    colors, halo_colors, mass_colors, gal_type = get_scene_colors(
        positions, stellar_mass, sfr, gal_type, halo_positions)

    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    radius = BOX_SIZE * 1.5

    for i in range(n_frames):
        # MPI: skip frames not assigned to this rank
        if i % size != rank:
            continue

        angle = 2 * np.pi * NUM_ORBITS * i / n_frames
        elevation = np.sin(angle * 0.5) * 30
        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + radius * np.sin(angle)
        cam_z = center[2] + radius * 0.5 * np.sin(elevation * np.pi / 180)

        plotter.clear_actors()
        if SHOW_HALOS and halo_positions is not None:
            add_halos_to_plotter(plotter, halo_positions, halo_masses, halo_colors)
        add_galaxies_to_plotter(plotter, positions, colors, sizes,
                                gal_type=gal_type, mass_colors=mass_colors)
        if SHOW_BOX: add_box_to_plotter(plotter)

        info = f'z = {redshift:.2f} | Mode: {COLOR_MODE.title()}'
        add_text_annotation(plotter, info)

        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        writer.write_frame(i)  # Pass frame index for correct naming
        if (i + 1) % 50 == 0:
            print(f"    Rank {rank}: Frame {i+1}/{n_frames}")

    mpi_barrier()  # Wait for all ranks to finish
    writer.close()
    plotter.close()


def find_density_peaks(positions, n_peaks=12, grid_size=10):
    """
    Find density peaks in the galaxy/halo distribution to use as flythrough waypoints.
    Uses a 3D grid to identify high-density regions, then returns their centers.
    Path is designed to traverse through high-density regions and end near box center.
    """
    if len(positions) < 100:
        # Fallback to simple waypoints if not enough data
        return None

    mpi_print("    Finding density peaks for flythrough path...")

    # Create 3D histogram to find density peaks
    bins = np.linspace(0, BOX_SIZE, grid_size + 1)
    hist, edges = np.histogramdd(positions, bins=[bins, bins, bins])

    # Find peak cells (local maxima or top N density cells)
    flat_indices = np.argsort(hist.flatten())[::-1]  # Sort by density, descending

    # Get unique peak locations (avoid adjacent cells)
    peaks = []
    peak_densities = []
    min_separation = BOX_SIZE / grid_size * 1.5  # Minimum separation between peaks

    for flat_idx in flat_indices:
        if len(peaks) >= n_peaks:
            break

        # Convert flat index to 3D indices
        ix, iy, iz = np.unravel_index(flat_idx, hist.shape)

        # Get cell center
        center = np.array([
            (edges[0][ix] + edges[0][ix+1]) / 2,
            (edges[1][iy] + edges[1][iy+1]) / 2,
            (edges[2][iz] + edges[2][iz+1]) / 2
        ])

        # Check separation from existing peaks
        if len(peaks) > 0:
            distances = np.linalg.norm(np.array(peaks) - center, axis=1)
            if np.min(distances) < min_separation:
                continue

        peaks.append(center)
        peak_densities.append(hist.flatten()[flat_idx])

    if len(peaks) < 4:
        return None

    peaks = np.array(peaks)
    peak_densities = np.array(peak_densities)
    box_center = np.array([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 2])

    # Find the peak closest to the box center - this will be our end point
    distances_to_center = np.linalg.norm(peaks - box_center, axis=1)
    end_idx = np.argmin(distances_to_center)

    # Find the peak furthest from center to start from (near edge but still in dense region)
    start_idx = np.argmax(distances_to_center)

    # Order peaks to create a path from outer dense region to center
    # Use density-weighted nearest neighbor: prefer denser regions when choosing next waypoint
    ordered_peaks = [peaks[start_idx]]
    remaining = list(range(len(peaks)))
    remaining.remove(start_idx)

    # Remove end_idx from remaining - we'll add it at the end
    if end_idx in remaining:
        remaining.remove(end_idx)

    # Density-weighted nearest neighbor ordering that favors high-density regions
    while remaining:
        current = ordered_peaks[-1]
        # Calculate distance to each remaining peak
        distances = np.array([np.linalg.norm(peaks[i] - current) for i in remaining])
        # Normalize distances (invert so closer = higher score)
        max_dist = np.max(distances) + 1e-10
        distance_scores = 1.0 - (distances / max_dist)

        # Get densities of remaining peaks
        densities = np.array([peak_densities[i] for i in remaining])
        max_dens = np.max(densities) + 1e-10
        density_scores = densities / max_dens

        # Combined score: favor nearby AND dense regions (weights adjustable)
        combined_scores = 0.4 * distance_scores + 0.6 * density_scores

        best_idx = remaining[np.argmax(combined_scores)]
        ordered_peaks.append(peaks[best_idx])
        remaining.remove(best_idx)

    # Add the center peak as the final destination
    ordered_peaks.append(peaks[end_idx])

    ordered_peaks = np.array(ordered_peaks)

    # Add entry point outside the box, aligned with the starting peak
    entry = np.array([
        ordered_peaks[0][0] + (ordered_peaks[0][0] - box_center[0]) * 0.5,
        ordered_peaks[0][1] + (ordered_peaks[0][1] - box_center[1]) * 0.5,
        ordered_peaks[0][2] + (ordered_peaks[0][2] - box_center[2]) * 0.5
    ])
    # Clamp entry point to be outside box but not too far
    entry = np.clip(entry, -BOX_SIZE * 0.3, BOX_SIZE * 1.3)

    # Final waypoint is near box center (the densest region near center)
    # No exit point outside box - we end at the center
    waypoints = np.vstack([[entry], ordered_peaks])

    mpi_print(f"    Found {len(peaks)} density peaks for flythrough path")
    mpi_print(f"    Path ends near box center at ({ordered_peaks[-1][0]:.1f}, {ordered_peaks[-1][1]:.1f}, {ordered_peaks[-1][2]:.1f})")
    return waypoints


def create_flythrough_animation(output_file, snapshot='Snap_63'):
    mpi_print(f"\nCreating flythrough animation ({COLOR_MODE}) for {snapshot}...")
    if MPI_ENABLED:
        mpi_print(f"  Running with {size} MPI ranks")
    n_frames = int(FLYTHROUGH_DURATION * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    positions, stellar_mass, sfr, gal_type = load_galaxy_data(DATA_FILE, snapshot)
    sizes = get_mass_sizes(stellar_mass)
    redshift = get_snapshot_redshift(snapshot)

    halo_positions, halo_masses = None, None
    if SHOW_HALOS:
        snap_num = int(snapshot.replace('Snap_', ''))
        halo_positions, halo_masses = load_halo_data(TREE_DIR, snap_num)

    colors, halo_colors, mass_colors, gal_type = get_scene_colors(
        positions, stellar_mass, sfr, gal_type, halo_positions)

    # Generate density-based waypoints from halo positions (preferred) or galaxy positions
    path_positions = halo_positions if halo_positions is not None and len(halo_positions) > 100 else positions
    waypoints = find_density_peaks(path_positions, n_peaks=12)

    if waypoints is None:
        # Fallback to default waypoints - path through interior ending at center
        mpi_print("    Using fallback waypoints (not enough density data)")
        waypoints = np.array([
            [-BOX_SIZE*0.2, BOX_SIZE*0.3, BOX_SIZE*0.3],   # Entry from corner
            [BOX_SIZE*0.2, BOX_SIZE*0.35, BOX_SIZE*0.35],  # Move into box
            [BOX_SIZE*0.35, BOX_SIZE*0.5, BOX_SIZE*0.45],  # Toward interior
            [BOX_SIZE*0.45, BOX_SIZE*0.6, BOX_SIZE*0.55],  # Through interior
            [BOX_SIZE*0.55, BOX_SIZE*0.55, BOX_SIZE*0.5],  # Near center region
            [BOX_SIZE*0.6, BOX_SIZE*0.45, BOX_SIZE*0.45],  # Continue through
            [BOX_SIZE*0.55, BOX_SIZE*0.5, BOX_SIZE*0.5],   # Approach center
            [BOX_SIZE*0.5, BOX_SIZE*0.5, BOX_SIZE*0.5],    # End at box center
        ])

    spline = pv.Spline(waypoints, 1000)
    path_points = spline.points
    look_ahead = 50

    for i in range(n_frames):
        # MPI: skip frames not assigned to this rank
        if i % size != rank:
            continue

        path_idx = int((i / n_frames) * (len(path_points) - look_ahead - 1))
        cam_pos = path_points[path_idx]
        look_idx = min(path_idx + look_ahead, len(path_points) - 1)
        focal_point = path_points[look_idx]

        plotter.clear_actors()
        if SHOW_HALOS and halo_positions is not None:
            add_halos_to_plotter(plotter, halo_positions, halo_masses, halo_colors)
        add_galaxies_to_plotter(plotter, positions, colors, sizes,
                                gal_type=gal_type, mass_colors=mass_colors)
        if SHOW_BOX: add_box_to_plotter(plotter)

        add_text_annotation(plotter, f'z = {redshift:.2f} | Mode: {COLOR_MODE.title()}')

        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal_point
        plotter.camera.up = (0, 0, 1)
        writer.write_frame(i)  # Pass frame index for correct naming
        if (i + 1) % 50 == 0:
            print(f"    Rank {rank}: Frame {i+1}/{n_frames}")

    mpi_barrier()  # Wait for all ranks to finish
    writer.close()
    plotter.close()


def create_evolution_animation(output_file, start_snap=30, end_snap=63):
    """
    Time evolution with smooth crossfade transitions between snapshots.
    Each frame smoothly blends between consecutive snapshots.
    """
    mpi_print(f"\nCreating time evolution ({COLOR_MODE}) with smooth transitions...")
    if MPI_ENABLED:
        mpi_print(f"  Running with {size} MPI ranks")
    n_frames = int(EVOLUTION_DURATION * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    # Pre-load all snapshot data for smooth interpolation
    mpi_print("  Pre-loading all snapshots for smooth transitions...")
    snapshot_data = {}
    for snap_idx in range(start_snap, end_snap + 1):
        snapshot = f'Snap_{snap_idx}'
        try:
            positions, stellar_mass, sfr, gal_type = load_galaxy_data(DATA_FILE, snapshot)
            sizes = get_mass_sizes(stellar_mass)
            redshift = get_snapshot_redshift(snapshot)

            halo_positions, halo_masses = None, None
            if SHOW_HALOS:
                try:
                    halo_positions, halo_masses = load_halo_data(TREE_DIR, snap_idx)
                except:
                    pass

            colors, halo_colors, mass_colors, gal_type = get_scene_colors(
                positions, stellar_mass, sfr, gal_type, halo_positions)

            snapshot_data[snap_idx] = {
                'positions': positions, 'stellar_mass': stellar_mass,
                'sizes': sizes, 'colors': colors, 'redshift': redshift,
                'halo_positions': halo_positions, 'halo_masses': halo_masses,
                'halo_colors': halo_colors, 'mass_colors': mass_colors, 'gal_type': gal_type
            }
        except Exception as e:
            mpi_print(f"    Warning: Could not load Snap_{snap_idx}: {e}")

    if len(snapshot_data) < 2:
        mpi_print("  ERROR: Need at least 2 snapshots for evolution animation")
        return

    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    cam_pos = np.array([BOX_SIZE*1.8, BOX_SIZE*1.8, BOX_SIZE*1.2])

    snap_indices = sorted(snapshot_data.keys())
    mpi_print(f"  Rendering {n_frames} frames across {len(snap_indices)} snapshots...")

    for frame_idx in range(n_frames):
        # MPI: skip frames not assigned to this rank
        if frame_idx % size != rank:
            continue

        # Calculate fractional position in snapshot sequence
        t = frame_idx / (n_frames - 1)  # 0 to 1
        snap_float = t * (len(snap_indices) - 1)
        snap_lo_idx = int(snap_float)
        snap_hi_idx = min(snap_lo_idx + 1, len(snap_indices) - 1)
        blend = snap_float - snap_lo_idx  # 0 to 1 within this transition

        snap_lo = snap_indices[snap_lo_idx]
        snap_hi = snap_indices[snap_hi_idx]
        data_lo = snapshot_data[snap_lo]
        data_hi = snapshot_data[snap_hi]

        # Interpolate redshift for display
        z_interp = data_lo['redshift'] * (1 - blend) + data_hi['redshift'] * blend

        plotter.clear_actors()

        # Render both snapshots with blended opacity for smooth transition
        if snap_lo == snap_hi or blend < 0.01:
            # Single snapshot (no blend needed)
            if SHOW_HALOS and data_lo['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_lo['halo_positions'],
                                     data_lo['halo_masses'], data_lo['halo_colors'])
            add_galaxies_to_plotter(plotter, data_lo['positions'],
                                    data_lo['colors'], data_lo['sizes'],
                                    gal_type=data_lo['gal_type'], mass_colors=data_lo['mass_colors'])
            n_gals = len(data_lo['positions'])
        elif blend > 0.99:
            # Single snapshot (blend complete)
            if SHOW_HALOS and data_hi['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_hi['halo_positions'],
                                     data_hi['halo_masses'], data_hi['halo_colors'])
            add_galaxies_to_plotter(plotter, data_hi['positions'],
                                    data_hi['colors'], data_hi['sizes'],
                                    gal_type=data_hi['gal_type'], mass_colors=data_hi['mass_colors'])
            n_gals = len(data_hi['positions'])
        else:
            # Crossfade: show both with blended opacity
            opacity_lo = 1.0 - blend
            opacity_hi = blend

            # Add fading-out snapshot (lower redshift)
            if SHOW_HALOS and data_lo['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_lo['halo_positions'],
                                     data_lo['halo_masses'], data_lo['halo_colors'],
                                     opacity_scale=opacity_lo)
            add_galaxies_to_plotter(plotter, data_lo['positions'],
                                    data_lo['colors'], data_lo['sizes'],
                                    opacity_scale=opacity_lo,
                                    gal_type=data_lo['gal_type'], mass_colors=data_lo['mass_colors'])

            # Add fading-in snapshot (higher redshift)
            if SHOW_HALOS and data_hi['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_hi['halo_positions'],
                                     data_hi['halo_masses'], data_hi['halo_colors'],
                                     opacity_scale=opacity_hi)
            add_galaxies_to_plotter(plotter, data_hi['positions'],
                                    data_hi['colors'], data_hi['sizes'],
                                    opacity_scale=opacity_hi,
                                    gal_type=data_hi['gal_type'], mass_colors=data_hi['mass_colors'])

            n_gals = int(len(data_lo['positions']) * opacity_lo + len(data_hi['positions']) * opacity_hi)

        if SHOW_BOX:
            add_box_to_plotter(plotter)

        add_text_annotation(plotter, f'z = {z_interp:.2f} | N ~ {n_gals}')
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        writer.write_frame(frame_idx)  # Pass frame index for correct naming

        if (frame_idx + 1) % 50 == 0:
            print(f"    Rank {rank}: Frame {frame_idx+1}/{n_frames} (z={z_interp:.2f})")

    mpi_barrier()  # Wait for all ranks to finish
    writer.close()
    plotter.close()
    mpi_print(f"  Completed: {output_file}")


def create_combined_animation(output_file, start_snap=30, end_snap=63):
    """
    Combined: time evolution with orbiting camera + smooth crossfade transitions.
    """
    mpi_print(f"\nCreating combined animation ({COLOR_MODE}) with smooth transitions...")
    if MPI_ENABLED:
        mpi_print(f"  Running with {size} MPI ranks")
    n_frames = int(COMBINED_DURATION * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    # Pre-load all snapshot data for smooth interpolation
    mpi_print("  Pre-loading all snapshots for smooth transitions...")
    snapshot_data = {}
    for snap_idx in range(start_snap, end_snap + 1):
        snapshot = f'Snap_{snap_idx}'
        try:
            positions, stellar_mass, sfr, gal_type = load_galaxy_data(DATA_FILE, snapshot)
            sizes = get_mass_sizes(stellar_mass)
            redshift = get_snapshot_redshift(snapshot)

            halo_positions, halo_masses = None, None
            if SHOW_HALOS:
                try:
                    halo_positions, halo_masses = load_halo_data(TREE_DIR, snap_idx)
                except:
                    pass

            colors, halo_colors, mass_colors, gal_type = get_scene_colors(
                positions, stellar_mass, sfr, gal_type, halo_positions)

            snapshot_data[snap_idx] = {
                'positions': positions, 'stellar_mass': stellar_mass,
                'sizes': sizes, 'colors': colors, 'redshift': redshift,
                'halo_positions': halo_positions, 'halo_masses': halo_masses,
                'halo_colors': halo_colors, 'mass_colors': mass_colors, 'gal_type': gal_type
            }
        except Exception as e:
            mpi_print(f"    Warning: Could not load Snap_{snap_idx}: {e}")

    if len(snapshot_data) < 2:
        mpi_print("  ERROR: Need at least 2 snapshots for combined animation")
        return

    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    radius = BOX_SIZE * 1.8

    snap_indices = sorted(snapshot_data.keys())
    mpi_print(f"  Rendering {n_frames} frames across {len(snap_indices)} snapshots...")

    for frame_idx in range(n_frames):
        # MPI: skip frames not assigned to this rank
        if frame_idx % size != rank:
            continue

        # Calculate fractional position in snapshot sequence
        t = frame_idx / (n_frames - 1)  # 0 to 1
        snap_float = t * (len(snap_indices) - 1)
        snap_lo_idx = int(snap_float)
        snap_hi_idx = min(snap_lo_idx + 1, len(snap_indices) - 1)
        blend = snap_float - snap_lo_idx

        snap_lo = snap_indices[snap_lo_idx]
        snap_hi = snap_indices[snap_hi_idx]
        data_lo = snapshot_data[snap_lo]
        data_hi = snapshot_data[snap_hi]

        # Interpolate redshift for display
        z_interp = data_lo['redshift'] * (1 - blend) + data_hi['redshift'] * blend

        # Camera orbit
        angle = 2 * np.pi * frame_idx / n_frames
        elevation = 30 + 15 * np.sin(angle * 2)
        cam_x = center[0] + radius * np.cos(angle) * np.cos(np.radians(elevation))
        cam_y = center[1] + radius * np.sin(angle) * np.cos(np.radians(elevation))
        cam_z = center[2] + radius * np.sin(np.radians(elevation))

        plotter.clear_actors()

        # Render with crossfade
        if snap_lo == snap_hi or blend < 0.01:
            if SHOW_HALOS and data_lo['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_lo['halo_positions'],
                                     data_lo['halo_masses'], data_lo['halo_colors'])
            add_galaxies_to_plotter(plotter, data_lo['positions'],
                                    data_lo['colors'], data_lo['sizes'],
                                    gal_type=data_lo['gal_type'], mass_colors=data_lo['mass_colors'])
        elif blend > 0.99:
            if SHOW_HALOS and data_hi['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_hi['halo_positions'],
                                     data_hi['halo_masses'], data_hi['halo_colors'])
            add_galaxies_to_plotter(plotter, data_hi['positions'],
                                    data_hi['colors'], data_hi['sizes'],
                                    gal_type=data_hi['gal_type'], mass_colors=data_hi['mass_colors'])
        else:
            # Crossfade between snapshots
            opacity_lo = 1.0 - blend
            opacity_hi = blend

            if SHOW_HALOS and data_lo['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_lo['halo_positions'],
                                     data_lo['halo_masses'], data_lo['halo_colors'],
                                     opacity_scale=opacity_lo)
            add_galaxies_to_plotter(plotter, data_lo['positions'],
                                    data_lo['colors'], data_lo['sizes'],
                                    opacity_scale=opacity_lo,
                                    gal_type=data_lo['gal_type'], mass_colors=data_lo['mass_colors'])

            if SHOW_HALOS and data_hi['halo_positions'] is not None:
                add_halos_to_plotter(plotter, data_hi['halo_positions'],
                                     data_hi['halo_masses'], data_hi['halo_colors'],
                                     opacity_scale=opacity_hi)
            add_galaxies_to_plotter(plotter, data_hi['positions'],
                                    data_hi['colors'], data_hi['sizes'],
                                    opacity_scale=opacity_hi,
                                    gal_type=data_hi['gal_type'], mass_colors=data_hi['mass_colors'])

        if SHOW_BOX:
            add_box_to_plotter(plotter)

        add_text_annotation(plotter, f'z = {z_interp:.2f}')

        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        writer.write_frame(frame_idx)  # Pass frame index for correct naming

        if (frame_idx + 1) % 50 == 0:
            print(f"    Rank {rank}: Frame {frame_idx+1}/{n_frames} (z={z_interp:.2f})")

    mpi_barrier()  # Wait for all ranks to finish
    writer.close()
    plotter.close()
    mpi_print(f"  Completed: {output_file}")


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(description='SAGE26 Simulation Flythrough Generator')
    parser.add_argument('--mode', type=str, default='orbit',
                        choices=['orbit', 'flythrough', 'evolution', 'combined', 'all'],
                        help='Animation mode')
    parser.add_argument('--color-by', type=str, default='mass',
                        choices=['mass', 'density', 'sfr', 'type'],
                        help='Color by: mass, density, sfr, or type (centrals/satellites)')
    parser.add_argument('--format', type=str, default='frames',
                        choices=['frames', 'gif', 'mp4', 'mov'],
                        help='Output format')
    parser.add_argument('--snapshot', type=str, default='Snap_63',
                        help='Snapshot for orbit/flythrough')
    parser.add_argument('--num-orbits', type=int, default=1,
                        help='Number of orbits (total time = orbit_duration * num_orbits)')
    parser.add_argument('--start-snap', type=int, default=30, help='Start snapshot')
    parser.add_argument('--end-snap', type=int, default=63, help='End snapshot')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output dir')
    parser.add_argument('--force', action='store_true', help='Force re-render')

    # Halo colormap arguments
    parser.add_argument('--halo-mass-cmap', type=str, default=None,
                        help='Colormap for halos in mass mode (default: plasma)')
    parser.add_argument('--halo-density-cmap', type=str, default=None,
                        help='Colormap for halos in density mode (default: magma)')
    parser.add_argument('--halo-sfr-cmap', type=str, default=None,
                        help='Colormap for halos in sSFR mode (default: plasma)')
    parser.add_argument('--halo-type-cmap', type=str, default=None,
                        help='Colormap for halos in type mode (default: plasma)')

    args = parser.parse_args()

    # Set Globals
    global OUTPUT_FORMAT, COLOR_MODE, NUM_ORBITS
    global HALO_MASS_COLORMAP, HALO_DENSITY_COLORMAP, HALO_SFR_COLORMAP, HALO_TYPE_COLORMAP
    OUTPUT_FORMAT = args.format
    COLOR_MODE = args.color_by
    NUM_ORBITS = args.num_orbits

    # Apply halo colormap overrides if specified
    if args.halo_mass_cmap:
        HALO_MASS_COLORMAP = args.halo_mass_cmap
    if args.halo_density_cmap:
        HALO_DENSITY_COLORMAP = args.halo_density_cmap
    if args.halo_sfr_cmap:
        HALO_SFR_COLORMAP = args.halo_sfr_cmap
    if args.halo_type_cmap:
        HALO_TYPE_COLORMAP = args.halo_type_cmap

    # Only rank 0 handles cleanup
    if args.force and rank == 0:
        import shutil
        # Clean up frame directories for all color modes
        suffix = f"_{args.color_by}" if args.color_by != 'mass' else ""
        for mode in ['orbit', 'flythrough', 'evolution', 'combined']:
            fdir = os.path.join(args.output_dir, f'sage26_{mode}{suffix}_frames')
            if os.path.exists(fdir): shutil.rmtree(fdir)

    mpi_barrier()  # Ensure cleanup is done before continuing

    os.makedirs(args.output_dir, exist_ok=True)

    mpi_print("=" * 60)
    mpi_print("SAGE26 Simulation Flythrough Generator")
    mpi_print(f"Mode: {args.mode}")
    mpi_print(f"Color By: {args.color_by.upper()}")
    if MPI_ENABLED:
        mpi_print(f"MPI Ranks: {size}")
    mpi_print("=" * 60)

    if not os.path.exists(DATA_FILE):
        mpi_print(f"ERROR: Data file not found: {DATA_FILE}")
        return

    modes_to_run = [args.mode] if args.mode != 'all' else ['orbit', 'flythrough', 'evolution', 'combined']
    ext = {'frames':'mp4', 'gif':'gif', 'mp4':'mp4', 'mov':'mov'}.get(OUTPUT_FORMAT, 'mp4')

    for mode in modes_to_run:
        # Append color mode to filename (except for default 'mass' mode)
        suffix = f"_{args.color_by}" if args.color_by != 'mass' else ""
        output_file = os.path.join(args.output_dir, f'sage26_{mode}{suffix}.{ext}')

        if mode == 'orbit':
            create_orbit_animation(output_file, args.snapshot)
        elif mode == 'flythrough':
            create_flythrough_animation(output_file, args.snapshot)
        elif mode == 'evolution':
            create_evolution_animation(output_file, args.start_snap, args.end_snap)
        elif mode == 'combined':
            create_combined_animation(output_file, args.start_snap, args.end_snap)

    mpi_print("\nDone!")

if __name__ == '__main__':
    main()
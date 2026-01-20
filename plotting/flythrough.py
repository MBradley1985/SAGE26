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

Author: Generated for SAGE26 SAM visualization
"""

import h5py as h5
import numpy as np
import pyvista as pv
import argparse
import os
from scipy.stats import gaussian_kde

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
ORBIT_DURATION = 60      # seconds for one full orbit
FLYTHROUGH_DURATION = 60  # seconds for flythrough (longer = smoother)
EVOLUTION_DURATION = 20   # seconds for time evolution
COMBINED_DURATION = 20    # seconds for combined animation

# Output format: 'mp4', 'mov', 'gif', or 'frames' (individual PNG files)
OUTPUT_FORMAT = 'frames'

# Galaxy selection
MIN_STELLAR_MASS = 1.0e8  # Minimum stellar mass in Msun (after unit conversion)
MAX_GALAXIES = 50000      # Maximum galaxies to render (for performance)

# Visual settings
POINT_SIZE = 3.0
BACKGROUND_COLOR = 'black'
BOX_COLOR = 'white'
BOX_OPACITY = 0.3
SHOW_BOX = False  # Set to True to show simulation box wireframe
GALAXY_OPACITY = 1.0  # Opacity for galaxy points (0.0 - 1.0)

# Particle size bins (can add more bins for finer gradation)
HALO_SIZE_BINS = [25.0, 30.0, 50.0, 70.0]  # Size bins for halos
GALAXY_SIZE_SCALE = 0.1  # Galaxy sizes = halo sizes * this factor

# Coloring Modes
COLOR_MODE = 'mass'     # Default mode: 'mass', 'density', 'sfr', or 'type'
MASS_COLORMAP = 'plasma'    # Colormap for Mass mode
DENSITY_COLORMAP = 'magma'  # Colormap for Density mode
SFR_COLORMAP = 'coolwarm_r'    # Colormap for sSFR mode (specific star formation rate)
CENTRAL_COLORMAP = 'Blues_r'   # Colormap for central galaxies (Type mode, colored by mass)
SATELLITE_COLORMAP = 'Reds_r'  # Colormap for satellite galaxies (Type mode, colored by mass)

# Colormap ranges (set to None for auto-scaling based on data)
# Values are in log10 units where applicable
STELLAR_MASS_RANGE = [8.0, 12.0]   # log10(Msun) range for mass coloring
SSFR_RANGE = [-14.0, -8.0]         # log10(yr^-1) range for sSFR coloring
DENSITY_RANGE = None               # Set to [min, max] to fix density range, or None for auto

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
    print(f"  Loading halos for snapshot {snapshot_num}...")

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
        print(f"    No halos found for snapshot {snapshot_num}")
        return np.array([]).reshape(0, 3), np.array([])

    positions = np.vstack(all_positions)
    masses = np.concatenate(all_masses)

    if len(positions) > max_halos:
        np.random.seed(42)
        idx = np.random.choice(len(positions), max_halos, replace=False)
        positions = positions[idx]
        masses = masses[idx]

    print(f"    Selected {len(positions)} halos")
    return positions, masses


def get_halo_sizes(masses):
    """Scale halo point sizes by mass using HALO_SIZE_BINS range."""
    if len(masses) == 0: return np.array([])
    min_size, max_size = HALO_SIZE_BINS[0], HALO_SIZE_BINS[-1]
    log_mass = np.log10(masses + 1)
    normalized = (log_mass - log_mass.min()) / (log_mass.max() - log_mass.min() + 1e-10)
    return min_size + normalized * (max_size - min_size)


def add_halos_to_plotter(plotter, positions, masses, colors=None, opacity_scale=1.0):
    """
    Add halo point cloud to the plotter.
    If 'colors' is None, defaults to mass-based coloring.
    opacity_scale: multiplier for opacity (used during crossfade transitions)
    """
    if len(positions) == 0:
        return

    sizes = get_halo_sizes(masses)

    # Determine coloring scheme - use same colormap as galaxies for consistency
    if colors is None:
        colors = np.log10(masses)
        cmap = MASS_COLORMAP  # Mass mode: same colormap as galaxies
    else:
        # Density mode: same colormap as galaxies
        cmap = DENSITY_COLORMAP

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
    print(f"  Loading {snapshot}...")
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
        print(f"    Warning: Missing field {e}")
        return np.array([]).reshape(0,3), np.array([]), np.array([]), np.array([])

    mask = (stellar_mass > mass_cut) & (mvir > 0)
    indices = np.where(mask)[0]

    if len(indices) > max_gals:
        np.random.seed(42)
        indices = np.random.choice(indices, max_gals, replace=False)

    positions = np.column_stack([posx[indices], posy[indices], posz[indices]])
    print(f"    Selected {len(indices)} galaxies")
    return positions, stellar_mass[indices], ssfr[indices], gal_type[indices]


def create_box_mesh():
    box = pv.Box(bounds=(0, BOX_SIZE, 0, BOX_SIZE, 0, BOX_SIZE))
    edges = box.extract_all_edges()
    return edges


def compute_density_colors(positions):
    """Compute KDE-based density coloring for galaxies/halos using DENSITY_RANGE."""
    if len(positions) < 10:
        return np.ones(len(positions))

    print("    Computing density estimates...")
    # Subsample for KDE computation to keep it fast
    if len(positions) > 5000:
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
        print(f"    Density computation failed: {e}")
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
    """Scale galaxy point sizes by mass using scaled HALO_SIZE_BINS range."""
    min_size = HALO_SIZE_BINS[0] * GALAXY_SIZE_SCALE
    max_size = HALO_SIZE_BINS[-1] * GALAXY_SIZE_SCALE
    log_mass = np.log10(stellar_mass + 1)
    normalized = (log_mass - log_mass.min()) / (log_mass.max() - log_mass.min() + 1e-10)
    return min_size + normalized * (max_size - min_size)


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
        print(f"  Found {len(existing)} existing frames in {frames_dir}/")
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
                print(f"  Skipping render - using existing frames")
            else:
                os.makedirs(self.frames_dir, exist_ok=True)
                print(f"  Saving frames to: {self.frames_dir}/")
        elif self.format == 'gif':
            import imageio
            self.writer = imageio.get_writer(output_path.replace('.mp4', '.gif'), mode='I', duration=1.0/fps)
        elif self.format == 'mp4':
            try:
                self.plotter.open_movie(output_path, framerate=fps, quality=8)
            except Exception as e:
                print(f"  Warning: MP4 creation failed ({e}), falling back to frames")
                self.format = 'frames'
                base = os.path.splitext(output_path)[0]
                self.frames_dir = base + '_frames'
                os.makedirs(self.frames_dir, exist_ok=True)

    def write_frame(self):
        if self.skip_rendering:
            self.frame_count += 1
            return

        if self.format in ('frames', 'mov'):
            frame_path = os.path.join(self.frames_dir, f'frame_{self.frame_count:05d}.png')
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

        ffmpeg_available = shutil.which('ffmpeg') is not None
        if self.format in ('frames', 'mov') and ffmpeg_available:
            print(f"\n  Converting frames with ffmpeg...")
            ext = '.mov' if self.format == 'mov' else '.mp4'
            out_path = self.output_path.replace('.mp4', ext) if self.format == 'mov' else self.output_path
            cmd = ['ffmpeg', '-y', '-framerate', str(self.fps),
                   '-i', f'{self.frames_dir}/frame_%05d.png',
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path]
            subprocess.run(cmd, capture_output=True)
            print(f"  Created: {out_path}")


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
    print(f"\nCreating orbit animation ({COLOR_MODE}) for {snapshot}...")
    n_frames = int(ORBIT_DURATION * FPS)
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
        angle = 2 * np.pi * i / n_frames
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
        writer.write_frame()
        if (i + 1) % 50 == 0: print(f"    Frame {i+1}/{n_frames}")

    writer.close()
    plotter.close()


def create_flythrough_animation(output_file, snapshot='Snap_63'):
    print(f"\nCreating flythrough animation ({COLOR_MODE}) for {snapshot}...")
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

    waypoints = np.array([
        [-BOX_SIZE*0.3, BOX_SIZE*0.5, BOX_SIZE*0.5],
        [BOX_SIZE*0.15, BOX_SIZE*0.45, BOX_SIZE*0.45],
        [BOX_SIZE*0.35, BOX_SIZE*0.35, BOX_SIZE*0.4],
        [BOX_SIZE*0.5, BOX_SIZE*0.5, BOX_SIZE*0.5],
        [BOX_SIZE*0.65, BOX_SIZE*0.65, BOX_SIZE*0.55],
        [BOX_SIZE*0.75, BOX_SIZE*0.5, BOX_SIZE*0.6],
        [BOX_SIZE*0.6, BOX_SIZE*0.35, BOX_SIZE*0.5],
        [BOX_SIZE*0.4, BOX_SIZE*0.4, BOX_SIZE*0.45],
        [BOX_SIZE*0.2, BOX_SIZE*0.5, BOX_SIZE*0.5],
    ])
    spline = pv.Spline(waypoints, 1000)
    path_points = spline.points
    look_ahead = 50

    for i in range(n_frames):
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
        writer.write_frame()
        if (i + 1) % 50 == 0: print(f"    Frame {i+1}/{n_frames}")

    writer.close()
    plotter.close()


def create_evolution_animation(output_file, start_snap=30, end_snap=63):
    """
    Time evolution with smooth crossfade transitions between snapshots.
    Each frame smoothly blends between consecutive snapshots.
    """
    print(f"\nCreating time evolution ({COLOR_MODE}) with smooth transitions...")
    total_snaps = end_snap - start_snap + 1
    n_frames = int(EVOLUTION_DURATION * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    # Pre-load all snapshot data for smooth interpolation
    print("  Pre-loading all snapshots for smooth transitions...")
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
            print(f"    Warning: Could not load Snap_{snap_idx}: {e}")

    if len(snapshot_data) < 2:
        print("  ERROR: Need at least 2 snapshots for evolution animation")
        return

    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    cam_pos = np.array([BOX_SIZE*1.8, BOX_SIZE*1.8, BOX_SIZE*1.2])

    snap_indices = sorted(snapshot_data.keys())
    print(f"  Rendering {n_frames} frames across {len(snap_indices)} snapshots...")

    for frame_idx in range(n_frames):
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
        writer.write_frame()

        if (frame_idx + 1) % 50 == 0:
            print(f"    Frame {frame_idx+1}/{n_frames} (z={z_interp:.2f})")

    writer.close()
    plotter.close()
    print(f"  Completed: {output_file}")


def create_combined_animation(output_file, start_snap=30, end_snap=63):
    """
    Combined: time evolution with orbiting camera + smooth crossfade transitions.
    """
    print(f"\nCreating combined animation ({COLOR_MODE}) with smooth transitions...")
    n_frames = int(COMBINED_DURATION * FPS)
    plotter = setup_plotter(off_screen=True)
    writer = FrameWriter(plotter, output_file, expected_frames=n_frames)

    if writer.skip_rendering:
        writer.close()
        return

    # Pre-load all snapshot data for smooth interpolation
    print("  Pre-loading all snapshots for smooth transitions...")
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
            print(f"    Warning: Could not load Snap_{snap_idx}: {e}")

    if len(snapshot_data) < 2:
        print("  ERROR: Need at least 2 snapshots for combined animation")
        return

    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    radius = BOX_SIZE * 1.8

    snap_indices = sorted(snapshot_data.keys())
    print(f"  Rendering {n_frames} frames across {len(snap_indices)} snapshots...")

    for frame_idx in range(n_frames):
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
        writer.write_frame()

        if (frame_idx + 1) % 50 == 0:
            print(f"    Frame {frame_idx+1}/{n_frames} (z={z_interp:.2f})")

    writer.close()
    plotter.close()
    print(f"  Completed: {output_file}")


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
    parser.add_argument('--start-snap', type=int, default=30, help='Start snapshot')
    parser.add_argument('--end-snap', type=int, default=63, help='End snapshot')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output dir')
    parser.add_argument('--force', action='store_true', help='Force re-render')

    args = parser.parse_args()

    # Set Globals
    global OUTPUT_FORMAT, COLOR_MODE
    OUTPUT_FORMAT = args.format
    COLOR_MODE = args.color_by

    if args.force:
        import shutil
        # Clean up frame directories for all color modes
        suffix = f"_{args.color_by}" if args.color_by != 'mass' else ""
        for mode in ['orbit', 'flythrough', 'evolution', 'combined']:
            fdir = os.path.join(args.output_dir, f'sage26_{mode}{suffix}_frames')
            if os.path.exists(fdir): shutil.rmtree(fdir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAGE26 Simulation Flythrough Generator")
    print(f"Mode: {args.mode}")
    print(f"Color By: {args.color_by.upper()}")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
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

    print("\nDone!")

if __name__ == '__main__':
    main()
#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import Delaunay
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import os

# ========================== USER OPTIONS ==========================

DirName = './output/millennium/'
FileName = 'model_0.hdf5'
SnapshotNum = 63

Hubble_h = 0.73
BoxSize = 62.5

# --- SETTINGS TO MIMIC GAS DENSITY ---
slice_thickness = 62.5  # Thick slice to get enough matter
grid_resolution = 4096   # Higher res for finer detail
smoothing_sigma = 1.0    # Less blur preserves filamentary structure
floor_value = 1e6        # Lower floor to show fainter structures

# --- FILAMENT STRAND SETTINGS ---
max_edge_length = 1.5    # Mpc/h - max length for Delaunay edges (filters long void-crossing edges)
subsample_factor = 1     # Use every Nth galaxy (1 = all, increase if too slow/dense)
line_alpha = 0.15        # Transparency of strands
line_width = 0.3         # Width of strands

OutputDir = DirName + 'plots/'
OutputFormat = '.png'

# ==================================================================

# --- EXACT UCHUU COLORMAP ---
uchuu_colors = [
    "#000000",  # Pure Black (Background)
    "#0a3d4d",  # dark teal 
    "#0f4d5e",  # deep teal
    "#15616d",  # medium teal
    "#1b7a8a",  # teal-cyan
    "#2a8f9f",  # bright teal
    "#4a9f8e",  # teal-green
    "#6aaa7a",  # yellow-green transition
    "#8ab566",  # olive-yellow
    "#b5a84f",  # golden
    "#d49a3a",  # orange-gold
    "#e88c2a",  # bright orange 
    "#f5a623",  # golden-orange 
    "#ffffff",  # White hot centers
]
uchuu_cmap = LinearSegmentedColormap.from_list("uchuu", uchuu_colors, N=1024)

# ==================================================================

def read_hdf(snap_num, param):
    try:
        f = h5.File(DirName+FileName,'r')
        return np.array(f[f'Snap_{snap_num}'][param])
    except Exception as e:
        print(f"Error reading {param}: {e}")
        return np.array([])

def get_redshift(snapshot_num):
    # Simplified dictionary for brevity
    shifts = {63: 0.0} 
    return shifts.get(snapshot_num, None)

if __name__ == '__main__':
    
    print('Processing...')
    if not os.path.exists(OutputDir): os.makedirs(OutputDir)
    
    # 1. Load Data
    Mvir = read_hdf(SnapshotNum, 'Mvir') * 1.0e10 / Hubble_h
    Posx = read_hdf(SnapshotNum, 'Posx')
    Posy = read_hdf(SnapshotNum, 'Posy')
    Posz = read_hdf(SnapshotNum, 'Posz')
    
    # 2. Slice Selection (Deep Projection)
    z_min = BoxSize/2 - slice_thickness/2
    z_max = BoxSize/2 + slice_thickness/2
    mask = (Posz >= z_min) & (Posz <= z_max)
    
    pos_x = Posx[mask]
    pos_y = Posy[mask]
    mass = Mvir[mask]
    
    print(f"Projecting {len(mass)} galaxies...")

    # 3. BUILD COSMIC WEB USING DELAUNAY TRIANGULATION
    # This connects nearby galaxies to form strand-like filaments

    # Subsample if needed (for performance)
    if subsample_factor > 1:
        idx = np.arange(0, len(pos_x), subsample_factor)
        pos_x_sub = pos_x[idx]
        pos_y_sub = pos_y[idx]
        mass_sub = mass[idx]
    else:
        pos_x_sub = pos_x
        pos_y_sub = pos_y
        mass_sub = mass

    print(f"Building Delaunay triangulation with {len(pos_x_sub)} points...")
    points = np.column_stack([pos_x_sub, pos_y_sub])
    tri = Delaunay(points)

    # Extract all edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            edges.add(edge)
    edges = np.array(list(edges))

    # Calculate edge lengths and filter out long edges (these cross voids)
    p1 = points[edges[:, 0]]
    p2 = points[edges[:, 1]]
    edge_lengths = np.sqrt(np.sum((p1 - p2)**2, axis=1))

    # Keep only short edges (these trace filaments)
    short_mask = edge_lengths < max_edge_length
    filtered_edges = edges[short_mask]
    filtered_lengths = edge_lengths[short_mask]

    print(f"Kept {len(filtered_edges)}/{len(edges)} edges (max length {max_edge_length} Mpc/h)")

    # Build line segments for plotting
    segments = []
    edge_masses = []  # For coloring by mass
    for edge in filtered_edges:
        i, j = edge
        segments.append([[points[i, 0], points[i, 1]],
                         [points[j, 0], points[j, 1]]])
        # Color by average mass of connected galaxies
        edge_masses.append((mass_sub[i] + mass_sub[j]) / 2)

    segments = np.array(segments)
    edge_masses = np.array(edge_masses)
    edge_masses_log = np.log10(edge_masses + 1e6)

    # 4. Also create density field for background glow
    H, xedges, yedges = np.histogram2d(
        pos_x,
        pos_y,
        bins=grid_resolution,
        range=[[0, BoxSize], [0, BoxSize]],
        weights=mass
    )

    # Log-transform and smooth for background
    H_log = np.log10(H + floor_value)
    H_smooth = gaussian_filter(H_log, sigma=smoothing_sigma * 3)  # More smoothing for glow

    # 5. Plotting
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111)

    # 5a. Background glow layer (smoothed density field)
    vmin_bg = np.percentile(H_smooth, 50)
    vmax_bg = np.percentile(H_smooth, 99.5)

    im = ax.imshow(
        H_smooth.T,
        origin='lower',
        extent=[0, BoxSize, 0, BoxSize],
        cmap=uchuu_cmap,
        aspect='auto',
        interpolation='bicubic',
        vmin=vmin_bg,
        vmax=vmax_bg,
        alpha=0.6  # Semi-transparent background
    )

    # 5b. STRAND LAYER - the actual cosmic web filaments
    # Normalize edge masses for coloring
    norm = plt.Normalize(vmin=np.percentile(edge_masses_log, 10),
                         vmax=np.percentile(edge_masses_log, 99))

    # Create line collection with colors based on mass
    lc = LineCollection(segments, cmap=uchuu_cmap, norm=norm,
                        linewidths=line_width, alpha=line_alpha)
    lc.set_array(edge_masses_log)
    ax.add_collection(lc)

    ax.set_xlim(0, BoxSize)
    ax.set_ylim(0, BoxSize)

    # Clean up axes like the snippet
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add Title
    redshift = get_redshift(SnapshotNum)
    if redshift:
        ax.set_title(f'z = {redshift:.2f}', fontsize=20, color='white', pad=20)

    # Add Colorbar (Horizontal bottom, like snippet)
    # cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, orientation="horizontal")
    # cb.set_label(r'$\log_{10}(\rho_{Mvir})$', color='white', fontsize=14)
    # cb.ax.xaxis.set_tick_params(color='white')
    # plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='white')

    output_filename = f'uchuu_GAS_STYLE_snap{SnapshotNum}' + OutputFormat
    plt.savefig(OutputDir + output_filename, dpi=300, bbox_inches='tight', facecolor='black')
    print(f'Saved to {OutputDir}{output_filename}')
    plt.close()
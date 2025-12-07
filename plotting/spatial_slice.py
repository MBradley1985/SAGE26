#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
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

    # 3. DTFE - Delaunay Tessellation Field Estimator
    # Estimates density at each galaxy, then interpolates across triangles
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    print("Building Delaunay tessellation for DTFE...")
    points = np.column_stack([pos_x, pos_y])
    tri = Delaunay(points)

    # Calculate area of each triangle
    def triangle_area(p1, p2, p3):
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

    # Calculate density at each point (sum of mass / sum of adjacent triangle areas)
    print("Calculating DTFE densities...")
    point_density = np.zeros(len(points))
    point_area = np.zeros(len(points))

    for simplex in tri.simplices:
        i, j, k = simplex
        area = triangle_area(points[i], points[j], points[k])
        # Each vertex gets 1/3 of the triangle area
        point_area[i] += area / 3
        point_area[j] += area / 3
        point_area[k] += area / 3

    # Density = mass / area (avoid division by zero)
    point_area = np.maximum(point_area, 1e-10)
    point_density = mass / point_area

    # Interpolate density field onto grid
    print("Interpolating density field...")
    grid_size = 16384
    x_grid = np.linspace(0, BoxSize, grid_size)
    y_grid = np.linspace(0, BoxSize, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Use linear interpolation within triangles
    # Take log of density BEFORE interpolating to reduce peaks
    log_density = np.log10(point_density + floor_value)
    interp = LinearNDInterpolator(points, log_density, fill_value=np.log10(floor_value))
    hist_log = interp(X, Y)

    # Smoothing for continuous look (scaled with grid size)
    hist_log = gaussian_filter(hist_log, sigma=200.0)

    # 4. Plotting
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    im = ax.imshow(hist_log, origin='lower', cmap=uchuu_cmap,
                   extent=[0, BoxSize, 0, BoxSize],
                   interpolation='bilinear',
                   vmin=np.percentile(hist_log, 2),
                   vmax=np.percentile(hist_log, 99.5))

    ax.set_xlim(0, BoxSize)
    ax.set_ylim(0, BoxSize)
    ax.set_aspect('equal')

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
#!/usr/bin/env python
"""
Validate the Boylan-Kolchin 2025 acceleration-based FFB regime determination.

Produces diagnostic plots and prints summary statistics to verify
that the BK25 method activates FFB at the correct redshifts/masses.

Usage:
    python plotting/validate_bk25_ffb.py
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ---------------------------------------------------------------------------
# Add project root so we can import plotting helpers if needed
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

# Try to use the same style as paper_plots
STYLE_FILE = os.path.join(SCRIPT_DIR, 'ciaran_ohare_palatino_sty.mplstyle')
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_bk25')
PRIMARY_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
# Conversion: g/G [M_sun/pc^2] -> g [m/s^2]
#   G * M_sun / pc^2 = 6.6743e-11 * 1.98892e30 / (3.08568e16)^2
G_NEWTON = 6.6743e-11       # m^3 kg^-1 s^-2
M_SUN_KG = 1.98892e30       # kg
PC_M     = 3.08568e16       # m
G_CONV   = G_NEWTON * M_SUN_KG / PC_M**2   # ~ 1.394e-13 m/s^2 per (M_sun/pc^2)

G_CRIT_OVER_G = 3100.0      # M_sun / pc^2  (BK25 Table 1)
G_CRIT        = 5.0e-10      # m/s^2  (BK25, ~ g_crit/G * G_CONV)


# ---------------------------------------------------------------------------
# I/O helpers (self-contained, no dependency on paper_plots)
# ---------------------------------------------------------------------------

def find_model_files(directory):
    import glob
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(glob.glob(pattern))
    return files


def read_header(directory):
    files = find_model_files(directory)
    if not files:
        raise FileNotFoundError(f"No model_*.hdf5 files in {directory}")
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        runtime = f['Header/Runtime']
        header = {
            'hubble_h': float(sim.attrs['hubble_h']),
            'box_size': float(sim.attrs['box_size']),
            'omega_matter': float(sim.attrs['omega_matter']),
            'omega_lambda': float(sim.attrs['omega_lambda']),
            'last_snap_nr': int(sim.attrs['LastSnapshotNr']),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'redshifts': list(f['Header/snapshot_redshifts'][:]),
            'output_snaps': list(f['Header/output_snapshots'][:]),
        }
    # Total volume fraction
    total_fvp = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    header['volume_fraction'] = total_fvp
    return header


def read_snap(directory, snap_num, properties):
    files = find_model_files(directory)
    snap_key = f'Snap_{snap_num}'
    chunks = {p: [] for p in properties}
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for p in properties:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))
    data = {}
    for p in properties:
        if chunks[p]:
            data[p] = np.concatenate(chunks[p])
    return data


# ---------------------------------------------------------------------------
# BK25 physics (pure python, mirrors the C code)
# ---------------------------------------------------------------------------

def mu_nfw(x):
    """NFW mu function: mu(x) = ln(1+x) - x/(1+x)"""
    return np.log(1.0 + x) - x / (1.0 + x)


# Ishiyama+21 concentration-mass lookup table (same as in C code)
_CM_LOGMASS = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0])
_CM_Z = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
_CM_TABLE = np.array([
    [19.444, 15.542, 12.268, 9.853, 8.119, 6.857, 5.922, 5.213, 4.667, 4.238, 3.898, 3.626, 3.407, 3.230, 3.087, 2.972, 2.879, 2.806, 2.748, 2.705, 2.673, 2.652, 2.639, 2.634, 2.636, 2.645, 2.659, 2.678, 2.701, 2.728, 2.760],
    [18.825, 15.041, 11.869, 9.533, 7.857, 6.640, 5.740, 5.059, 4.535, 4.126, 3.803, 3.545, 3.339, 3.173, 3.040, 2.934, 2.850, 2.785, 2.735, 2.698, 2.674, 2.659, 2.652, 2.654, 2.662, 2.676, 2.695, 2.720, 2.748, 2.781, 2.818],
    [18.213, 14.545, 11.475, 9.216, 7.599, 6.427, 5.562, 4.909, 4.408, 4.018, 3.711, 3.468, 3.274, 3.120, 2.997, 2.901, 2.826, 2.769, 2.727, 2.697, 2.679, 2.671, 2.671, 2.679, 2.693, 2.713, 2.739, 2.769, 2.803, 2.842, 2.884],
    [17.606, 14.053, 11.084, 8.904, 7.345, 6.217, 5.387, 4.762, 4.284, 3.913, 3.623, 3.394, 3.214, 3.071, 2.959, 2.872, 2.806, 2.757, 2.723, 2.702, 2.691, 2.690, 2.697, 2.711, 2.732, 2.758, 2.790, 2.826, 2.867, 2.911, 2.959],
    [17.003, 13.566, 10.698, 8.595, 7.094, 6.011, 5.216, 4.619, 4.164, 3.813, 3.539, 3.325, 3.158, 3.027, 2.926, 2.849, 2.792, 2.752, 2.726, 2.712, 2.709, 2.716, 2.730, 2.751, 2.778, 2.811, 2.849, 2.892, 2.939, 2.989, 3.043],
    [16.405, 13.083, 10.315, 8.290, 6.848, 5.809, 5.048, 4.479, 4.048, 3.716, 3.460, 3.261, 3.106, 2.988, 2.897, 2.831, 2.783, 2.752, 2.735, 2.730, 2.735, 2.748, 2.770, 2.798, 2.833, 2.873, 2.918, 2.967, 3.021, 3.078, 3.139],
    [15.812, 12.605, 9.937, 7.989, 6.605, 5.610, 4.885, 4.344, 3.936, 3.624, 3.385, 3.201, 3.060, 2.954, 2.875, 2.818, 2.781, 2.759, 2.751, 2.754, 2.767, 2.789, 2.819, 2.855, 2.897, 2.944, 2.997, 3.054, 3.114, 3.179, 3.247],
    [15.225, 12.132, 9.564, 7.693, 6.367, 5.417, 4.726, 4.214, 3.829, 3.537, 3.315, 3.147, 3.020, 2.926, 2.858, 2.813, 2.786, 2.774, 2.775, 2.787, 2.809, 2.839, 2.877, 2.921, 2.971, 3.027, 3.087, 3.152, 3.220, 3.292, 3.367],
    [14.649, 11.669, 9.200, 7.404, 6.135, 5.229, 4.573, 4.089, 3.728, 3.457, 3.252, 3.099, 2.986, 2.905, 2.850, 2.815, 2.799, 2.797, 2.808, 2.829, 2.860, 2.900, 2.946, 2.999, 3.058, 3.122, 3.190, 3.263, 3.340, 3.420, 3.504],
    [14.083, 11.214, 8.842, 7.122, 5.910, 5.048, 4.427, 3.971, 3.634, 3.382, 3.196, 3.058, 2.960, 2.892, 2.849, 2.826, 2.821, 2.830, 2.851, 2.882, 2.923, 2.972, 3.028, 3.090, 3.157, 3.230, 3.308, 3.389, 3.475, 3.564, 3.656],
    [13.525, 10.767, 8.492, 6.847, 5.691, 4.873, 4.286, 3.859, 3.545, 3.315, 3.146, 3.025, 2.941, 2.887, 2.857, 2.846, 2.852, 2.872, 2.904, 2.946, 2.997, 3.056, 3.122, 3.194, 3.272, 3.354, 3.441, 3.533, 3.628, 3.726, 3.828],
    [12.976, 10.327, 8.149, 6.577, 5.479, 4.704, 4.152, 3.754, 3.464, 3.254, 3.104, 3.000, 2.931, 2.891, 2.874, 2.876, 2.895, 2.927, 2.970, 3.023, 3.085, 3.155, 3.232, 3.314, 3.402, 3.495, 3.593, 3.695, 3.800, 3.909, 4.021],
    [12.434, 9.894, 7.812, 6.315, 5.273, 4.542, 4.025, 3.655, 3.390, 3.202, 3.070, 2.983, 2.930, 2.905, 2.902, 2.918, 2.949, 2.994, 3.049, 3.115, 3.189, 3.270, 3.358, 3.452, 3.552, 3.656, 3.765, 3.878, 3.995, 4.115, 4.238],
    [11.898, 9.468, 7.481, 6.058, 5.073, 4.386, 3.905, 3.564, 3.324, 3.157, 3.046, 2.976, 2.940, 2.930, 2.942, 2.972, 3.017, 3.075, 3.144, 3.222, 3.309, 3.403, 3.504, 3.610, 3.722, 3.839, 3.960, 4.085, 4.215, 4.347, 4.483],
    [11.370, 9.049, 7.157, 5.809, 4.880, 4.238, 3.792, 3.481, 3.267, 3.122, 3.031, 2.980, 2.961, 2.968, 2.996, 3.041, 3.101, 3.173, 3.256, 3.348, 3.449, 3.557, 3.671, 3.791, 3.916, 4.047, 4.181, 4.320, 4.463, 4.609, 4.758],
    [10.849, 8.636, 6.840, 5.566, 4.694, 4.097, 3.687, 3.407, 3.219, 3.098, 3.027, 2.996, 2.996, 3.020, 3.064, 3.126, 3.201, 3.289, 3.387, 3.495, 3.610, 3.733, 3.863, 3.998, 4.138, 4.283, 4.432, 4.586, 4.743, 4.904, 5.068],
    [10.335, 8.230, 6.530, 5.331, 4.517, 3.964, 3.592, 3.342, 3.181, 3.084, 3.036, 3.025, 3.044, 3.087, 3.150, 3.229, 3.322, 3.426, 3.541, 3.665, 3.797, 3.937, 4.082, 4.234, 4.390, 4.551, 4.717, 4.887, 5.061, 5.238, 5.418],
    [9.830, 7.834, 6.229, 5.105, 4.348, 3.842, 3.507, 3.289, 3.156, 3.085, 3.060, 3.071, 3.110, 3.173, 3.255, 3.353, 3.465, 3.588, 3.721, 3.863, 4.013, 4.171, 4.334, 4.504, 4.678, 4.857, 5.041, 5.229, 5.420, 5.615, 5.814],
    [9.339, 7.451, 5.940, 4.890, 4.192, 3.731, 3.435, 3.250, 3.146, 3.100, 3.100, 3.134, 3.196, 3.281, 3.384, 3.502, 3.634, 3.778, 3.931, 4.093, 4.263, 4.440, 4.624, 4.813, 5.007, 5.206, 5.410, 5.618, 5.829, 6.045, 6.263],
    [8.862, 7.080, 5.663, 4.688, 4.047, 3.634, 3.376, 3.225, 3.151, 3.134, 3.160, 3.219, 3.305, 3.413, 3.539, 3.680, 3.834, 4.000, 4.175, 4.359, 4.552, 4.751, 4.956, 5.167, 5.384, 5.605, 5.831, 6.061, 6.295, 6.533, 6.774],
    [8.397, 6.721, 5.399, 4.498, 3.916, 3.550, 3.333, 3.217, 3.175, 3.187, 3.241, 3.327, 3.439, 3.573, 3.724, 3.890, 4.069, 4.260, 4.460, 4.668, 4.885, 5.109, 5.339, 5.575, 5.816, 6.062, 6.313, 6.568, 6.827, 7.090, 7.356],
    [7.944, 6.374, 5.146, 4.320, 3.798, 3.482, 3.306, 3.228, 3.220, 3.264, 3.348, 3.463, 3.604, 3.766, 3.945, 4.139, 4.345, 4.563, 4.791, 5.027, 5.271, 5.523, 5.780, 6.044, 6.313, 6.587, 6.866, 7.149, 7.436, 7.727, 8.022],
    [7.503, 6.040, 4.907, 4.157, 3.696, 3.430, 3.299, 3.260, 3.288, 3.367, 3.484, 3.632, 3.804, 3.997, 4.207, 4.432, 4.670, 4.918, 5.176, 5.444, 5.719, 6.001, 6.290, 6.585, 6.886, 7.191, 7.502, 7.817, 8.136, 8.459, 8.785],
    [7.076, 5.719, 4.682, 4.010, 3.611, 3.398, 3.313, 3.317, 3.385, 3.501, 3.655, 3.838, 4.046, 4.274, 4.518, 4.778, 5.050, 5.333, 5.627, 5.929, 6.239, 6.557, 6.881, 7.211, 7.547, 7.889, 8.235, 8.586, 8.941, 9.301, 9.664],
    [6.662, 5.413, 4.472, 3.880, 3.546, 3.389, 3.353, 3.402, 3.513, 3.671, 3.865, 4.089, 4.336, 4.603, 4.887, 5.186, 5.497, 5.820, 6.153, 6.494, 6.845, 7.202, 7.567, 7.938, 8.314, 8.696, 9.083, 9.475, 9.872, 10.273, 10.677],
    [6.263, 5.123, 4.281, 3.770, 3.504, 3.406, 3.424, 3.522, 3.681, 3.885, 4.124, 4.392, 4.684, 4.996, 5.324, 5.667, 6.024, 6.391, 6.769, 7.156, 7.552, 7.955, 8.366, 8.782, 9.205, 9.634, 10.068, 10.507, 10.951, 11.399, 11.851],
    [5.886, 4.854, 4.112, 3.685, 3.490, 3.455, 3.530, 3.684, 3.895, 4.150, 4.441, 4.759, 5.102, 5.464, 5.843, 6.238, 6.645, 7.064, 7.494, 7.933, 8.381, 8.837, 9.300, 9.770, 10.246, 10.728, 11.216, 11.710, 12.208, 12.711, 13.218],
    [5.531, 4.610, 3.970, 3.629, 3.510, 3.542, 3.681, 3.894, 4.165, 4.479, 4.827, 5.204, 5.604, 6.024, 6.462, 6.915, 7.381, 7.860, 8.349, 8.848, 9.357, 9.874, 10.398, 10.930, 11.468, 12.012, 12.562, 13.118, 13.680, 14.246, 14.817],
    [5.201, 4.391, 3.855, 3.605, 3.567, 3.674, 3.883, 4.165, 4.502, 4.882, 5.297, 5.740, 6.208, 6.695, 7.201, 7.722, 8.257, 8.804, 9.363, 9.932, 10.511, 11.099, 11.694, 12.298, 12.908, 13.524, 14.147, 14.777, 15.412, 16.052, 16.697],
    [4.896, 4.200, 3.773, 3.621, 3.670, 3.859, 4.147, 4.507, 4.921, 5.378, 5.870, 6.391, 6.936, 7.502, 8.086, 8.687, 9.302, 9.930, 10.570, 11.221, 11.882, 12.553, 13.233, 13.920, 14.614, 15.316, 16.025, 16.740, 17.461, 18.188, 18.920],
    [4.618, 4.041, 3.729, 3.681, 3.828, 4.109, 4.488, 4.937, 5.440, 5.987, 6.569, 7.180, 7.816, 8.474, 9.151, 9.846, 10.556, 11.279, 12.015, 12.762, 13.521, 14.290, 15.068, 15.854, 16.648, 17.450, 18.259, 19.076, 19.900, 20.729, 21.564],
    [4.370, 3.918, 3.729, 3.797, 4.053, 4.440, 4.923, 5.475, 6.083, 6.735, 7.423, 8.141, 8.885, 9.652, 10.439, 11.245, 12.067, 12.903, 13.753, 14.615, 15.489, 16.374, 17.269, 18.173, 19.085, 20.006, 20.935, 21.873, 22.817, 23.768, 24.725],
    [4.158, 3.838, 3.784, 3.980, 4.360, 4.870, 5.474, 6.149, 6.880, 7.657, 8.470, 9.315, 10.189, 11.086, 12.004, 12.942, 13.898, 14.869, 15.854, 16.853, 17.865, 18.889, 19.924, 20.968, 22.023, 23.086, 24.158, 25.239, 26.328, 27.425, 28.528],
    [3.988, 3.811, 3.905, 4.247, 4.771, 5.424, 6.172, 6.993, 7.871, 8.796, 9.761, 10.759, 11.787, 12.840, 13.916, 15.013, 16.130, 17.263, 18.412, 19.576, 20.753, 21.945, 23.148, 24.362, 25.586, 26.821, 28.065, 29.320, 30.583, 31.854, 33.133],
    [3.869, 3.848, 4.110, 4.620, 5.313, 6.135, 7.056, 8.051, 9.107, 10.212, 11.359, 12.542, 13.757, 14.999, 16.266, 17.557, 18.869, 20.199, 21.546, 22.909, 24.289, 25.683, 27.090, 28.510, 29.941, 31.383, 32.837, 34.302, 35.777, 37.260, 38.752],
    [3.812, 3.966, 4.419, 5.127, 6.021, 7.048, 8.177, 9.384, 10.655, 11.980, 13.349, 14.757, 16.201, 17.675, 19.176, 20.702, 22.252, 23.823, 25.413, 27.021, 28.646, 30.289, 31.946, 33.616, 35.300, 36.997, 38.706, 40.428, 42.161, 43.905, 45.658],
    [3.830, 4.184, 4.863, 5.806, 6.942, 8.218, 9.602, 11.069, 12.605, 14.199, 15.843, 17.529, 19.254, 21.013, 22.801, 24.619, 26.463, 28.330, 30.218, 32.127, 34.056, 36.005, 37.970, 39.950, 41.945, 43.954, 45.979, 48.018, 50.070, 52.133, 54.207],
    [3.943, 4.531, 5.478, 6.707, 8.141, 9.723, 11.421, 13.211, 15.076, 17.005, 18.988, 21.020, 23.095, 25.207, 27.353, 29.532, 31.741, 33.977, 36.236, 38.519, 40.826, 43.154, 45.501, 47.866, 50.248, 52.646, 55.062, 57.495, 59.942, 62.403, 64.876],
    [4.176, 5.044, 6.317, 7.898, 9.701, 11.665, 13.758, 15.950, 18.227, 20.576, 22.985, 25.449, 27.962, 30.518, 33.112, 35.744, 38.411, 41.108, 43.832, 46.584, 49.363, 52.167, 54.993, 57.840, 60.706, 63.592, 66.498, 69.424, 72.367, 75.325, 78.299],
    [4.564, 5.773, 7.450, 9.471, 11.737, 14.183, 16.772, 19.473, 22.269, 25.147, 28.094, 31.103, 34.169, 37.284, 40.445, 43.648, 46.892, 50.171, 53.482, 56.825, 60.200, 63.604, 67.034, 70.488, 73.966, 77.466, 80.990, 84.538, 88.106, 91.691, 95.295],
    [5.150, 6.782, 8.965, 11.539, 14.391, 17.445, 20.662, 24.006, 27.459, 31.006, 34.633, 38.331, 42.096, 45.918, 49.792, 53.717, 57.690, 61.703, 65.755, 69.843, 73.969, 78.131, 82.323, 86.543, 90.791, 95.066, 99.369, 103.700, 108.055, 112.431, 116.829]
])
from scipy.interpolate import RegularGridInterpolator
_CM_INTERP = RegularGridInterpolator((_CM_LOGMASS, _CM_Z), _CM_TABLE,
                                      method='linear', bounds_error=False, fill_value=None)

def ishiyama21_concentration(logM, z):
    """Ishiyama+21 c(M,z) via bilinear interpolation of the lookup table.
    logM is log10(Mvir / [Msun/h]), z is redshift. Both can be arrays."""
    logM = np.clip(np.atleast_1d(logM), _CM_LOGMASS[0], _CM_LOGMASS[-1])
    z = np.clip(np.atleast_1d(z), _CM_Z[0], _CM_Z[-1])
    pts = np.column_stack([logM.ravel(), z.ravel()])
    c = _CM_INTERP(pts).reshape(logM.shape)
    return np.maximum(c, 1.0)



def ishiyama21_threshold_mvir(z_arr, h=0.73):
    """Solve for the FFB threshold Mvir at each redshift using Ishiyama+21 c(M,z)
    and BK25 Eq. 5 approximation for Rvir.
    Returns array of Mvir in Msun (not Msun/h)."""
    from scipy.optimize import brentq
    R_coeff = 7.0e3  # Rvir in pc when Mvir=1e10 Msun, (1+z)/10=1  (BK25 Eq. 5)
    result = np.full_like(z_arr, np.nan)
    for i, z in enumerate(z_arr):
        def g_max_minus_gcrit(log_mvir):
            Mvir_Msun = 10**log_mvir
            Rvir_pc = R_coeff * (Mvir_Msun / 1e10)**(1.0/3.0) * ((1 + z) / 10.0)**(-1)
            g_vir_over_G = Mvir_Msun / Rvir_pc**2
            logM_h = np.log10(Mvir_Msun * h)
            c = ishiyama21_concentration(logM_h, z).item()
            mc = np.log(1 + c) - c / (1 + c)
            g_max = g_vir_over_G * c**2 / (2 * mc)
            return g_max - G_CRIT_OVER_G
        try:
            log_m_thresh = brentq(g_max_minus_gcrit, 8, 14)
            result[i] = 10**log_m_thresh
        except ValueError:
            pass
    return result


def vmax_vvir_to_concentration(ratio):
    """
    Numerically invert (Vmax/Vvir)^2 = 0.216 * c / mu(c) to get c.
    Vectorised over an array of ratios.
    """
    ratio_sq = ratio**2
    c = 5.0 * ratio_sq  # initial guess
    for _ in range(30):
        mc = mu_nfw(c)
        f = 0.216 * c / mc - ratio_sq
        dmu = 1.0 / (1.0 + c) - 1.0 / (1.0 + c)**2
        df = 0.216 * (mc - c * dmu) / mc**2
        df = np.where(np.abs(df) < 1e-30, 1.0, df)
        c = c - f / df
        c = np.clip(c, 1.0, None)
    return c


def compute_g_max_over_G(Mvir_code, Rvir_code, Vmax, Vvir, h):
    """
    Compute g_max / G in M_sun / pc^2  (BK25 Eq. 4).
    Inputs in SAGE code units.
    """
    Mvir_Msun = Mvir_code * 1e10 / h
    Rvir_pc = Rvir_code * 1e6 / h

    g_vir_over_G = np.where(Rvir_pc > 0, Mvir_Msun / Rvir_pc**2, 0.0)

    # Concentration from Vmax/Vvir
    ratio = np.where(Vvir > 0, Vmax / Vvir, 1.0)
    c = vmax_vvir_to_concentration(ratio)

    mc = mu_nfw(c)
    g_max_over_G = g_vir_over_G * c**2 / (2.0 * mc)
    return g_max_over_G, g_vir_over_G, c


# ---------------------------------------------------------------------------
# Plot 1: BK25 Figure 1 (left) — g/g_vir vs r/R_vir for different c
# ---------------------------------------------------------------------------

def plot_nfw_acceleration_profiles():
    """Reproduce BK25 Fig 1 left panel: g(r)/g_vir vs r/R_vir."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    r_tilde = np.logspace(-3, 0, 500)  # r / R_vir
    concentrations = [2, 4, 6, 8, 10, 15, 20]
    cmap = plt.cm.viridis_r
    colors = [cmap(i / (len(concentrations) - 1)) for i in range(len(concentrations))]

    for c, color in zip(concentrations, colors):
        mc = mu_nfw(c)
        g_ratio = mu_nfw(c * r_tilde) / (mc * r_tilde**2)
        ax.plot(r_tilde, g_ratio, color=color, lw=2, label=f'c = {c}')

    # Mark g_crit / g_vir for some reference g_vir values
    ax.axhline(1.0, color='grey', ls=':', lw=0.8, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r / R_{\rm vir}$')
    ax.set_ylabel(r'$g(r) / g_{\rm vir}$')
    ax.set_title('NFW Acceleration Profile (BK25 Fig. 1 left)')
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0.5, 300)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_fig1_left_nfw_profiles.pdf'))
    print(f"  Saved: bk25_fig1_left_nfw_profiles.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: g_max/G vs M_vir at z=0 and z=10 (sanity check)
# ---------------------------------------------------------------------------

def _theory_gmax_over_G(Mvir, z):
    """Compute g_max/G [M_sun/pc^2] for halo mass array using BK25 Eq. 5 + Duffy."""
    Rvir_kpc = 7.0 * (Mvir / 1e10)**(1.0/3.0) * ((1 + z) / 10.0)**(-1)
    Rvir_pc = Rvir_kpc * 1e3
    g_vir_over_G = Mvir / Rvir_pc**2
    c = 7.85 * (Mvir / 2e12)**(-0.081) * (1 + z)**(-0.71)
    c = np.clip(c, 1.0, None)
    mc = mu_nfw(c)
    return g_vir_over_G * c**2 / (2.0 * mc)   # M_sun / pc^2


def plot_gmax_vs_mass_theory():
    """Show g_max/G vs M_vir for different redshifts — pure theory curves."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    log_Mvir = np.linspace(8, 15, 200)
    Mvir = 10**log_Mvir

    redshifts = [0, 2, 5, 8, 10, 14]
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(redshifts) - 1)) for i in range(len(redshifts))]

    for z, color in zip(redshifts, colors):
        g_max = _theory_gmax_over_G(Mvir, z)
        ax.plot(log_Mvir, np.log10(g_max), color=color, lw=2, label=f'z = {z}')

    ax.axhline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
               label=r'$g_{\rm crit}/G = 3100\;M_\odot\,{\rm pc}^{-2}$')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
    ax.set_title(r'$g_{\rm max}/G$ vs $M_{\rm vir}$ (Duffy+08 concentrations)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(8, 15)
    ax.set_ylim(-1, 5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_mass_theory_duffy.pdf'))
    print(f"  Saved: bk25_gmax_vs_mass_theory_duffy.pdf")
    plt.close(fig)


def plot_gmax_vs_stellar_mass_theory():
    """Show g_max vs M_star for different redshifts — theory curves."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    log_Mvir = np.linspace(8, 15, 300)
    Mvir = 10**log_Mvir

    def mstar_from_mvir(Mvir, z):
        f_b = 0.156
        log_M1 = 11.5 + 0.3 * np.minimum(z, 4)
        epsilon_peak = 0.2 * (1 + z)**(-0.3)
        x = np.log10(Mvir) - log_M1
        epsilon = epsilon_peak * 2.0 / (10**(0.5 * x) + 10**(-0.5 * x))
        return epsilon * f_b * Mvir

    redshifts_plot = [0, 2, 5, 8, 10, 14]
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(redshifts_plot) - 1)) for i in range(len(redshifts_plot))]

    for z, color in zip(redshifts_plot, colors):
        g_max = _theory_gmax_over_G(Mvir, z)
        Mstar = mstar_from_mvir(Mvir, z)
        valid = Mstar > 0
        ax.plot(np.log10(Mstar[valid]), np.log10(g_max[valid]),
                color=color, lw=2, label=f'z = {z}')

    ax.axhline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
               label=r'$g_{\rm crit}/G = 3100\;M_\odot\,{\rm pc}^{-2}$')
    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
    ax.set_title(r'$g_{\rm max}/G$ vs $M_\star$ (Duffy+08 conc., approx. SMHM)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(4, 13)
    ax.set_ylim(-1, 5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_stellar_mass_theory_duffy.pdf'))
    print(f"  Saved: bk25_gmax_vs_stellar_mass_theory_duffy.pdf")
    plt.close(fig)


def plot_gmax_vs_mass_theory_vmax_vvir(header):
    """
    g_max/G vs Mvir and vs M* using MEDIAN Vmax/Vvir from actual SAGE output
    binned by mass and redshift, instead of Duffy.
    """
    h = header['hubble_h']
    redshifts_all = header['redshifts']
    output_snaps = sorted(header['output_snaps'])
    mass_convert = header['unit_mass_in_g'] / 1.989e33 / h

    properties = ['Mvir', 'Rvir', 'Vmax', 'Vvir', 'StellarMass']

    target_zs = [0, 2, 5, 8, 10, 14]

    # Find closest output snap for each target z
    snap_for_z = {}
    for zt in target_zs:
        best_snap = min(output_snaps, key=lambda s: abs(redshifts_all[s] - zt))
        if abs(redshifts_all[best_snap] - zt) < 1.0:
            snap_for_z[zt] = best_snap

    cmap = plt.cm.plasma
    colors = {zt: cmap(i / max(len(target_zs) - 1, 1))
              for i, zt in enumerate(target_zs)}

    # --- Plot: g_max/G vs Mvir (lines = redshifts) ---
    fig_mvir, ax_mvir = plt.subplots(figsize=(7, 5.5))

    for zt in target_zs:
        if zt not in snap_for_z:
            continue
        snap = snap_for_z[zt]
        z_actual = redshifts_all[snap]
        data = read_snap(MODEL_DIR, snap, properties)
        if not data or 'Mvir' not in data:
            continue

        Mvir = data['Mvir']
        Rvir = data['Rvir']
        Vmax = data['Vmax']
        Vvir = data['Vvir']

        valid = (Mvir > 0) & (Rvir > 0) & (Vvir > 0) & (Vmax > 0)
        if not np.any(valid):
            continue

        g_max, g_vir, c = compute_g_max_over_G(
            Mvir[valid], Rvir[valid], Vmax[valid], Vvir[valid], h)
        Mvir_Msun = Mvir[valid] * mass_convert

        log_gmax = np.log10(np.clip(g_max, 1e-20, None))
        log_mvir = np.log10(Mvir_Msun)

        bins = np.linspace(log_mvir.min(), log_mvir.max(), 30)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        bin_idx = np.digitize(log_mvir, bins)
        medians_mvir = []
        valid_bins_mvir = []
        for bi in range(1, len(bins)):
            mask = bin_idx == bi
            if mask.sum() >= 5:
                medians_mvir.append(np.median(log_gmax[mask]))
                valid_bins_mvir.append(bin_centres[bi - 1])

        if valid_bins_mvir:
            ax_mvir.plot(valid_bins_mvir, medians_mvir, color=colors[zt], lw=2.5,
                         label=f'z = {z_actual:.1f}')

    ax_mvir.axhline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
                    label=r'$g_{\rm crit}/G$')
    ax_mvir.legend(fontsize=9, loc='lower right')
    ax_mvir.set_ylabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
    ax_mvir.set_ylim(-1, 5)
    ax_mvir.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax_mvir.set_title(r'$g_{\rm max}/G$ vs $M_{\rm vir}$ (Vmax/Vvir concentrations)')
    ax_mvir.set_xlim(8, 14)

    fig_mvir.tight_layout()
    fig_mvir.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_mvir_vmax_vvir.pdf'))
    print(f"  Saved: bk25_gmax_vs_mvir_vmax_vvir.pdf")
    plt.close(fig_mvir)

    # --- Plot: g_max/G vs M* at z=0 (lines = halo mass bins) ---
    # Find the z~0 snapshot
    snap_z0 = min(output_snaps, key=lambda s: abs(redshifts_all[s]))

    fig_mstar, ax_mstar = plt.subplots(figsize=(7, 5.5))

    data_z0 = read_snap(MODEL_DIR, snap_z0, properties)
    if data_z0 and 'Mvir' in data_z0:
        Mvir = data_z0['Mvir']
        Rvir = data_z0['Rvir']
        Vmax = data_z0['Vmax']
        Vvir = data_z0['Vvir']
        Mstar = data_z0['StellarMass'] * mass_convert if 'StellarMass' in data_z0 else np.zeros_like(Mvir)

        valid = (Mvir > 0) & (Rvir > 0) & (Vvir > 0) & (Vmax > 0) & (Mstar > 0)

        if np.any(valid):
            g_max, _, _ = compute_g_max_over_G(
                Mvir[valid], Rvir[valid], Vmax[valid], Vvir[valid], h)
            log_mvir_all = np.log10(Mvir[valid] * mass_convert)
            log_mstar_all = np.log10(Mstar[valid])

            mvir_bins = [(9, 10), (10, 10.5), (10.5, 11), (11, 11.5), (11.5, 12), (12, 13)]
            cmap_m = plt.cm.viridis
            colors_m = [cmap_m(i / max(len(mvir_bins) - 1, 1)) for i in range(len(mvir_bins))]

            for (mlo, mhi), color in zip(mvir_bins, colors_m):
                in_bin = (log_mvir_all >= mlo) & (log_mvir_all < mhi)
                if in_bin.sum() < 10:
                    continue

                ms_bin = log_mstar_all[in_bin]
                gm_bin = g_max[in_bin]

                bins_s = np.linspace(max(ms_bin.min(), 5), ms_bin.max(), 30)
                bin_centres_s = 0.5 * (bins_s[:-1] + bins_s[1:])
                bin_idx_s = np.digitize(ms_bin, bins_s)
                medians = []
                valid_bins = []
                for bi in range(1, len(bins_s)):
                    mask = bin_idx_s == bi
                    if mask.sum() >= 5:
                        medians.append(np.median(gm_bin[mask]))
                        valid_bins.append(bin_centres_s[bi - 1])

                if valid_bins:
                    ax_mstar.plot(valid_bins, medians, color=color, lw=2.5,
                                  label=rf'$\log M_{{\rm vir}} = {mlo}$–${mhi}$')

    ax_mstar.axhline(G_CRIT_OVER_G, color='red', ls='--', lw=2,
                     label=r'$g_{\rm crit}/G$')
    ax_mstar.legend(fontsize=9, loc='lower right')
    ax_mstar.set_yscale('log')
    ax_mstar.set_ylabel(r'$g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}]$')
    ax_mstar.set_ylim(1e-1, 1e5)
    ax_mstar.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    ax_mstar.set_title(r'$g_{\rm max}/G$ vs $M_\star$ at $z=0$ (Vmax/Vvir, by halo mass)')
    ax_mstar.set_xlim(5, 13)

    fig_mstar.tight_layout()
    fig_mstar.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_mstar_vmax_vvir.pdf'))
    print(f"  Saved: bk25_gmax_vs_mstar_vmax_vvir.pdf")
    plt.close(fig_mstar)


# ---------------------------------------------------------------------------
# Plots from actual SAGE output
# ---------------------------------------------------------------------------

def plot_ffb_diagnostics(header):
    """
    Load multiple snapshots and produce diagnostic plots:
      - FFB fraction vs redshift
      - g_max/G distribution at different redshifts
      - FFB galaxies in the Mvir-z plane
      - Concentration distribution for FFB vs non-FFB
    """
    h = header['hubble_h']
    redshifts = header['redshifts']
    output_snaps = sorted(header['output_snaps'])
    mass_convert = header['unit_mass_in_g'] / 1.989e33 / h

    properties = ['Mvir', 'Rvir', 'Vmax', 'Vvir', 'FFBRegime', 'Regime',
                  'StellarMass', 'Type', 'SfrDisk', 'SfrBulge']

    # Collect per-snapshot statistics
    snap_z = []
    snap_n_total = []
    snap_n_ffb = []
    snap_frac_ffb = []
    snap_median_gmax_ffb = []
    snap_median_gmax_all = []
    snap_median_mvir_ffb = []
    snap_median_c_ffb = []

    # Store detailed data for a few key snapshots
    detail_snaps = {}  # {snap: data_dict}

    print("\n--- Per-snapshot FFB diagnostics ---")
    print(f"{'Snap':>5s} {'z':>7s} {'N_total':>9s} {'N_FFB':>7s} {'f_FFB':>8s} "
          f"{'med gmax/G':>12s} {'med Mvir':>12s} {'med c':>7s}")

    for snap in output_snaps:
        z = redshifts[snap]
        data = read_snap(MODEL_DIR, snap, properties)
        if not data or 'Mvir' not in data:
            continue

        Mvir = data['Mvir']
        Rvir = data['Rvir']
        Vmax = data['Vmax']
        Vvir = data['Vvir']

        # Only consider resolved galaxies
        valid = (Mvir > 0) & (Rvir > 0) & (Vvir > 0)
        if not np.any(valid):
            continue

        g_max, g_vir, c = compute_g_max_over_G(
            Mvir[valid], Rvir[valid], Vmax[valid], Vvir[valid], h)

        n_total = int(valid.sum())

        if 'FFBRegime' in data:
            ffb = data['FFBRegime'][valid] == 1
        else:
            ffb = g_max > G_CRIT_OVER_G

        n_ffb = int(ffb.sum())
        frac = n_ffb / n_total if n_total > 0 else 0.0

        snap_z.append(z)
        snap_n_total.append(n_total)
        snap_n_ffb.append(n_ffb)
        snap_frac_ffb.append(frac)

        if n_ffb > 0:
            Mvir_Msun = Mvir[valid] * mass_convert
            snap_median_gmax_ffb.append(np.median(g_max[ffb]))
            snap_median_mvir_ffb.append(np.median(Mvir_Msun[ffb]))
            snap_median_c_ffb.append(np.median(c[ffb]))
        else:
            snap_median_gmax_ffb.append(np.nan)
            snap_median_mvir_ffb.append(np.nan)
            snap_median_c_ffb.append(np.nan)
        snap_median_gmax_all.append(np.median(g_max))

        # Store detail for interesting snapshots
        # z ~ 0, 2, 5, 8, 10, 14
        for zt in [0, 2, 5, 8, 10, 14]:
            if abs(z - zt) < 0.3 and zt not in detail_snaps:
                Mstar_Msun = data['StellarMass'][valid] * mass_convert if 'StellarMass' in data else np.zeros(valid.sum())
                detail_snaps[zt] = {
                    'z': z, 'snap': snap,
                    'g_max': g_max, 'g_vir': g_vir, 'c': c,
                    'Mvir_Msun': Mvir[valid] * mass_convert,
                    'Mstar_Msun': Mstar_Msun,
                    'ffb': ffb,
                }

        med_gmax_str = f"{snap_median_gmax_ffb[-1]:.1f}" if n_ffb > 0 else "---"
        med_mvir_str = f"{snap_median_mvir_ffb[-1]:.2e}" if n_ffb > 0 else "---"
        med_c_str = f"{snap_median_c_ffb[-1]:.1f}" if n_ffb > 0 else "---"
        print(f"{snap:5d} {z:7.3f} {n_total:9d} {n_ffb:7d} {frac:8.4f} "
              f"{med_gmax_str:>12s} {med_mvir_str:>12s} {med_c_str:>7s}")

    snap_z = np.array(snap_z)
    snap_n_ffb = np.array(snap_n_ffb)
    snap_frac_ffb = np.array(snap_frac_ffb)

    # ---- Plot A: FFB fraction vs redshift ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(snap_z, snap_n_ffb, 'o-', color='C0', ms=5)
    ax1.set_ylabel('Number of FFB galaxies')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.5)
    ax1.set_title('FFB Galaxy Count vs Redshift (BK25 method)')
    ax1.axhline(1, color='grey', ls=':', lw=0.8)

    ax2.plot(snap_z, snap_frac_ffb * 100, 'o-', color='C1', ms=5)
    ax2.set_ylabel('FFB fraction [%]')
    ax2.set_xlabel('Redshift')
    ax2.set_xlim(-0.5, max(snap_z) + 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_ffb_fraction_vs_z.pdf'))
    print(f"\n  Saved: bk25_ffb_fraction_vs_z.pdf")
    plt.close(fig)

    # ---- Plot B: g_max distributions at key redshifts ----
    if detail_snaps:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        target_zs = [0, 2, 5, 8, 10, 14]

        for ax, zt in zip(axes.flat, target_zs):
            if zt in detail_snaps:
                dd = detail_snaps[zt]
                log_gmax = np.log10(np.clip(dd['g_max'], 1e-20, None))
                ax.hist(log_gmax, bins=50, color='C0', alpha=0.7, label='All')
                if np.any(dd['ffb']):
                    ax.hist(log_gmax[dd['ffb']], bins=50, color='C3', alpha=0.7,
                            label='FFB')
                ax.axvline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
                           label=r'$g_{\rm crit}/G$')
                ax.set_title(f"z = {dd['z']:.2f} (snap {dd['snap']})")
            else:
                ax.text(0.5, 0.5, f'z={zt}\nno data', transform=ax.transAxes,
                        ha='center', va='center')
            ax.set_xlabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        fig.suptitle(r'Distribution of $g_{\rm max}/G$ at key redshifts', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_distributions.pdf'))
        print(f"  Saved: bk25_gmax_distributions.pdf")
        plt.close(fig)

    # ---- Plot C: FFB galaxies in Mvir-z plane ----
    if detail_snaps:
        fig, ax = plt.subplots(figsize=(8, 6))
        for zt in sorted(detail_snaps.keys()):
            dd = detail_snaps[zt]
            z_arr = np.full(len(dd['Mvir_Msun']), dd['z'])
            log_mvir = np.log10(dd['Mvir_Msun'])

            # Plot all as background
            ax.scatter(z_arr[~dd['ffb']], log_mvir[~dd['ffb']],
                       c='grey', s=1, alpha=0.05, rasterized=True)
            if np.any(dd['ffb']):
                ax.scatter(z_arr[dd['ffb']], log_mvir[dd['ffb']],
                           c='C3', s=8, alpha=0.6, zorder=5, rasterized=True)

        # Theory curve: M_vir at g_thresh (BK25 Eq 8, g_vir/G = 380)
        z_theory = np.linspace(4, 20, 100)
        g_thresh_over_G = 380.0  # M_sun / pc^2
        # From Eq. 8: Mvir = 1e10 * (g/204)^3 * ((1+z)/10)^-6
        Mvir_thresh = 1e10 * (g_thresh_over_G / 204.0)**3 * ((1 + z_theory) / 10.0)**(-6)
        ax.plot(z_theory, np.log10(Mvir_thresh), 'b--', lw=2,
                label=r'$g_{\rm vir}/G = 380\;(g_{\rm thresh})$')

        ax.set_xlabel('Redshift')
        ax.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
        ax.set_title('FFB galaxies (red) in the $M_{\\rm vir}$-$z$ plane')
        ax.legend()
        ax.set_xlim(-0.5, 16)
        ax.set_ylim(7.5, 14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_ffb_mvir_z_plane.pdf'))
        print(f"  Saved: bk25_ffb_mvir_z_plane.pdf")
        plt.close(fig)

    # ---- Plot D: Concentration distributions FFB vs non-FFB ----
    if detail_snaps:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        target_zs = [0, 2, 5, 8, 10, 14]

        for ax, zt in zip(axes.flat, target_zs):
            if zt in detail_snaps:
                dd = detail_snaps[zt]
                c_all = np.clip(dd['c'], 1, 100)
                ax.hist(c_all, bins=np.linspace(0, 40, 50), color='C0',
                        alpha=0.7, label='All', density=True)
                if np.any(dd['ffb']):
                    ax.hist(c_all[dd['ffb']], bins=np.linspace(0, 40, 50),
                            color='C3', alpha=0.7, label='FFB', density=True)
                ax.set_title(f"z = {dd['z']:.2f}")
            else:
                ax.text(0.5, 0.5, f'z={zt}\nno data', transform=ax.transAxes,
                        ha='center', va='center')
            ax.set_xlabel('Concentration c')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

        fig.suptitle('Concentration distribution: FFB vs All', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_concentration_distributions.pdf'))
        print(f"  Saved: bk25_concentration_distributions.pdf")
        plt.close(fig)

    # ---- Plot E: g_max vs Mvir scatter at z~0 and z~10 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, zt, title in [(ax1, 0, 'z = 0'), (ax2, 10, 'z = 10')]:
        if zt in detail_snaps:
            dd = detail_snaps[zt]
            log_mvir = np.log10(dd['Mvir_Msun'])
            log_gmax = np.log10(np.clip(dd['g_max'], 1e-20, None))

            n = len(log_mvir)
            idx = np.random.choice(n, min(n, 10000), replace=False)

            ax.scatter(log_mvir[idx], log_gmax[idx], c='C0', s=2, alpha=0.2,
                       rasterized=True, label='All galaxies')
            ffb_idx = idx[dd['ffb'][idx]]
            if len(ffb_idx) > 0:
                ax.scatter(log_mvir[ffb_idx], log_gmax[ffb_idx], c='C3', s=8,
                           alpha=0.6, zorder=5, label='FFB')
            ax.axhline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
                       label=r'$g_{\rm crit}/G$')

        ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
        ax.set_ylabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim(8, 14)
        ax.set_ylim(-1, 5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_mvir_scatter.pdf'))
    print(f"  Saved: bk25_gmax_vs_mvir_scatter.pdf")
    plt.close(fig)

    # ---- Plot F: g_max vs Stellar Mass scatter at key redshifts ----
    target_zs_stellar = [0, 2, 5, 8, 10, 14]
    available = [zt for zt in target_zs_stellar if zt in detail_snaps]
    ncols = min(len(available), 3)
    nrows = (len(available) + ncols - 1) // ncols if available else 1

    if available:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                                 squeeze=False)
        for i, zt in enumerate(available):
            ax = axes.flat[i]
            dd = detail_snaps[zt]

            has_stars = dd['Mstar_Msun'] > 0
            if not np.any(has_stars):
                ax.text(0.5, 0.5, f'z={zt}\nno stellar mass', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            log_mstar = np.log10(dd['Mstar_Msun'][has_stars])
            log_gmax = np.log10(np.clip(dd['g_max'][has_stars], 1e-20, None))
            ffb_s = dd['ffb'][has_stars]

            n = len(log_mstar)
            idx = np.random.choice(n, min(n, 10000), replace=False)

            ax.scatter(log_mstar[idx], log_gmax[idx], c='C0', s=2, alpha=0.15,
                       rasterized=True, label='All')
            ffb_idx = idx[ffb_s[idx]]
            if len(ffb_idx) > 0:
                ax.scatter(log_mstar[ffb_idx], log_gmax[ffb_idx], c='C3', s=10,
                           alpha=0.6, zorder=5, label='FFB')
            ax.axhline(np.log10(G_CRIT_OVER_G), color='red', ls='--', lw=2,
                       label=r'$g_{\rm crit}/G$')

            ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
            ax.set_ylabel(r'$\log_{10}(g_{\rm max}/G\;[M_\odot\,{\rm pc}^{-2}])$')
            ax.set_title(f'z = {dd["z"]:.2f}')
            ax.legend(fontsize=8, loc='lower right')
            ax.set_xlim(5, 13)
            ax.set_ylim(-1, 5)

        for j in range(len(available), nrows * ncols):
            axes.flat[j].set_visible(False)

        fig.suptitle(r'$g_{\rm max}/G$ vs Stellar Mass at key redshifts', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_gmax_vs_stellar_mass.pdf'))
        print(f"  Saved: bk25_gmax_vs_stellar_mass.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Mvir vs z color-coded by concentration, panels for different c values
# ---------------------------------------------------------------------------

def plot_mvir_z_by_concentration(header):
    """
    Multi-panel plot of Mvir vs z, with galaxies color-coded by their
    Ishiyama+21 concentration. Panel 1 uses the self-consistent Ishiyama+21
    threshold line, panels 2-5 show fixed-c reference lines, and the last
    two panels show Li+24 and BK25+Li+24 comparison.
    """
    h = header['hubble_h']
    redshifts_all = header['redshifts']
    output_snaps = sorted(header['output_snaps'])
    mass_convert = header['unit_mass_in_g'] / 1.989e33 / h

    # Try to read Concentration from output; fall back to Vmax/Vvir
    properties_try = ['Mvir', 'Rvir', 'Vmax', 'Vvir', 'FFBRegime', 'Concentration']

    # Collect data from all snapshots
    all_z = []
    all_log_mvir = []
    all_c = []
    all_ffb = []

    for snap in output_snaps:
        z = redshifts_all[snap]
        data = read_snap(MODEL_DIR, snap, properties_try)
        if not data or 'Mvir' not in data:
            continue

        Mvir = data['Mvir']
        Rvir = data['Rvir']
        Vmax = data['Vmax']
        Vvir = data['Vvir']

        valid = (Mvir > 0) & (Rvir > 0) & (Vvir > 0) & (Vmax > 0)
        if not np.any(valid):
            continue

        # Use Ishiyama+21 concentration from output if available
        if 'Concentration' in data and data['Concentration'] is not None:
            c = data['Concentration'][valid]
            c = np.where(c > 0, c, 1.0)
        else:
            _, _, c = compute_g_max_over_G(
                Mvir[valid], Rvir[valid], Vmax[valid], Vvir[valid], h)

        log_mvir = np.log10(Mvir[valid] * mass_convert)
        ffb = data['FFBRegime'][valid] == 1 if 'FFBRegime' in data else np.zeros(valid.sum(), dtype=bool)

        all_z.append(np.full(valid.sum(), z))
        all_log_mvir.append(log_mvir)
        all_c.append(c)
        all_ffb.append(ffb)

    if not all_z:
        print("  No data for Mvir-z concentration plot")
        return

    all_z = np.concatenate(all_z)
    all_log_mvir = np.concatenate(all_log_mvir)
    all_c = np.concatenate(all_c)
    all_ffb = np.concatenate(all_ffb)

    # Fixed-c threshold for reference panels
    def threshold_mvir_fixed_c(z_arr, c_val):
        mc = mu_nfw(c_val)
        factor = c_val**2 / (2.0 * mc)
        R_coeff = 7.0e3  # Rvir in pc when Mvir=1e10, (1+z)/10=1
        mvir_onethird = G_CRIT_OVER_G * R_coeff**2 / (1e10**(2.0/3.0) * ((1 + z_arr) / 10.0)**2 * factor)
        return mvir_onethird**3

    # Ishiyama+21 self-consistent threshold
    z_theory = np.linspace(0.01, 15, 200)
    Mvir_ish_thresh = ishiyama21_threshold_mvir(z_theory)
    log_ish_thresh = np.log10(np.where(Mvir_ish_thresh > 0, Mvir_ish_thresh, np.nan))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # Subsample all galaxies once for background scatter
    n_all = len(all_z)
    bg_idx = np.random.choice(n_all, min(n_all, 30000), replace=False) if n_all > 30000 else np.arange(n_all)
    ffb_idx_all = np.where(all_ffb)[0]

    # --- Panel 1: BK25 with Ishiyama+21 self-consistent threshold ---
    ax0 = axes.flat[0]
    ax0.scatter(all_z[bg_idx], all_log_mvir[bg_idx], c=all_c[bg_idx],
                s=2, alpha=0.15, cmap='viridis', vmin=1, vmax=15,
                rasterized=True)
    if len(ffb_idx_all) > 0:
        ax0.scatter(all_z[ffb_idx_all], all_log_mvir[ffb_idx_all], c='red',
                    s=12, alpha=0.7, zorder=5, label='FFB')
    valid_t = np.isfinite(log_ish_thresh) & (log_ish_thresh > 7) & (log_ish_thresh < 15)
    ax0.plot(z_theory[valid_t], log_ish_thresh[valid_t], 'k-', lw=3, zorder=10,
             label=r'$g_{\rm max} = g_{\rm crit}$ (Ishiyama+21)')
    ax0.set_title('BK25: Ishiyama+21 c(M,z)', fontsize=10)
    ax0.set_xlabel('Redshift')
    ax0.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax0.set_xlim(-0.5, 15)
    ax0.set_ylim(10, 14)
    ax0.legend(fontsize=7, loc='upper right')

    # --- Panels 2-6: Fixed-c reference lines ---
    c_panels = [3, 5, 8, 10, 15]
    for ax, c_target in zip(axes.flat[1:6], c_panels):
        ax.scatter(all_z[bg_idx], all_log_mvir[bg_idx], c=all_c[bg_idx],
                   s=2, alpha=0.15, cmap='viridis', vmin=1, vmax=15,
                   rasterized=True)
        if len(ffb_idx_all) > 0:
            ax.scatter(all_z[ffb_idx_all], all_log_mvir[ffb_idx_all], c='red',
                       s=12, alpha=0.7, zorder=5, label='FFB')

        Mvir_thresh = threshold_mvir_fixed_c(z_theory, c_target)
        log_thresh = np.log10(Mvir_thresh)
        valid_t = (log_thresh > 7) & (log_thresh < 15)
        ax.plot(z_theory[valid_t], log_thresh[valid_t], 'k-', lw=3, zorder=10,
                label=r'$g_{\rm max} = g_{\rm crit}$')

        # Also overlay the Ishiyama+21 line for comparison (dashed)
        valid_ish = np.isfinite(log_ish_thresh) & (log_ish_thresh > 7) & (log_ish_thresh < 15)
        ax.plot(z_theory[valid_ish], log_ish_thresh[valid_ish], 'k--', lw=1.5, alpha=0.5, zorder=9)

        ax.set_title(rf'BK25: fixed $c = {c_target}$', fontsize=10)
        ax.set_xlabel('Redshift')
        ax.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
        ax.set_xlim(-0.5, 15)
        ax.set_ylim(10, 14)
        ax.legend(fontsize=7, loc='upper right')

    # --- Panel 7: Li+24 mass-based threshold ---
    ax_li = axes.flat[6]
    li_z = []
    li_log_mvir = []
    li_c = []
    li_ffb = []

    li_properties = ['Mvir', 'Rvir', 'Vmax', 'Vvir', 'FFBRegime']
    li_files = find_model_files(PRIMARY_DIR)
    if li_files:
        li_header = read_header(PRIMARY_DIR)
        li_output_snaps = sorted(li_header['output_snaps'])
        li_redshifts = li_header['redshifts']
        li_mass_convert = li_header['unit_mass_in_g'] / 1.989e33 / li_header['hubble_h']
        li_h = li_header['hubble_h']

        for snap in li_output_snaps:
            z_snap = li_redshifts[snap]
            data = read_snap(PRIMARY_DIR, snap, li_properties)
            if not data or 'Mvir' not in data:
                continue

            Mvir = data['Mvir']
            Rvir = data['Rvir']
            Vmax = data['Vmax']
            Vvir = data['Vvir']

            valid = (Mvir > 0) & (Rvir > 0) & (Vvir > 0) & (Vmax > 0)
            if not np.any(valid):
                continue

            _, _, c_li = compute_g_max_over_G(
                Mvir[valid], Rvir[valid], Vmax[valid], Vvir[valid], li_h)

            ffb = data['FFBRegime'][valid] == 1 if 'FFBRegime' in data else np.zeros(valid.sum(), dtype=bool)

            li_z.append(np.full(valid.sum(), z_snap))
            li_log_mvir.append(np.log10(Mvir[valid] * li_mass_convert))
            li_c.append(c_li)
            li_ffb.append(ffb)

    if li_z:
        li_z = np.concatenate(li_z)
        li_log_mvir = np.concatenate(li_log_mvir)
        li_c = np.concatenate(li_c)
        li_ffb = np.concatenate(li_ffb)

        n_li = len(li_z)
        li_bg_idx = np.random.choice(n_li, min(n_li, 30000), replace=False) if n_li > 30000 else np.arange(n_li)
        li_ffb_idx = np.where(li_ffb)[0]

        ax_li.scatter(li_z[li_bg_idx], li_log_mvir[li_bg_idx], c=li_c[li_bg_idx],
                      s=2, alpha=0.15, cmap='viridis', vmin=1, vmax=30,
                      rasterized=True)
        if len(li_ffb_idx) > 0:
            ax_li.scatter(li_z[li_ffb_idx], li_log_mvir[li_ffb_idx], c='red',
                         s=12, alpha=0.7, zorder=5, label='FFB (Li+24)')
    else:
        ax_li.text(0.5, 0.5, f'No data\n{PRIMARY_DIR}', transform=ax_li.transAxes,
                   ha='center', va='center')

    # Li+24 threshold: M_vir,ffb = 10^10.8 * ((1+z)/10)^-6.2 M_sun
    z_li_theory = np.linspace(0.01, 15, 200)
    log_Mvir_li = 10.8 - 6.2 * np.log10((1 + z_li_theory) / 10.0)
    ax_li.plot(z_li_theory, log_Mvir_li, 'k-', lw=3, zorder=10, label=r'Li+24 $M_{\rm vir,ffb}$')

    ax_li.set_title('Li+24 (mass-based)', fontsize=10)
    ax_li.set_xlabel('Redshift')
    ax_li.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax_li.set_xlim(-0.5, 15)
    ax_li.set_ylim(10, 14)
    ax_li.legend(fontsize=7, loc='upper right')

    # --- Panel 8: Comparison of Li+24 and BK25 threshold lines ---
    ax_cmp = axes.flat[7]
    ax_cmp.scatter(all_z[bg_idx], all_log_mvir[bg_idx], c='grey',
                   s=1, alpha=0.1, rasterized=True)
    if len(ffb_idx_all) > 0:
        ax_cmp.scatter(all_z[ffb_idx_all], all_log_mvir[ffb_idx_all], c='red',
                       s=8, alpha=0.5, zorder=5, label='FFB (BK25)')
    valid_ish = np.isfinite(log_ish_thresh) & (log_ish_thresh > 7) & (log_ish_thresh < 15)
    ax_cmp.plot(z_theory[valid_ish], log_ish_thresh[valid_ish], 'k-', lw=3, zorder=10,
                label='BK25 (Ishiyama+21)')
    ax_cmp.plot(z_li_theory, log_Mvir_li, 'b--', lw=2.5, zorder=10, label='Li+24')
    ax_cmp.set_title('Threshold comparison', fontsize=10)
    ax_cmp.set_xlabel('Redshift')
    ax_cmp.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax_cmp.set_xlim(-0.5, 15)
    ax_cmp.set_ylim(10, 14)
    ax_cmp.legend(fontsize=7, loc='upper right')

    fig.suptitle(r'$M_{\rm vir}$ vs $z$ — FFB threshold comparison (colour = $c$ from Ishiyama+21)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bk25_mvir_z_by_concentration.pdf'))
    print(f"  Saved: bk25_mvir_z_by_concentration.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: Sanity check — MW and cluster at z=0
# ---------------------------------------------------------------------------

def print_sanity_checks():
    """Print g_max/G for known reference haloes to verify equations."""
    h = 0.73  # Millennium

    print("\n--- Sanity checks (BK25 reference values) ---\n")

    cases = [
        ("MW (z=0)", 1e12, 250e3, 10.0, 0),         # M_sun, R_vir in pc, c, z
        ("Cluster (z=0)", 1e15, 2.5e6, 5.0, 0),
        ("10^10.8 (z=10)", 10**10.8, 11.8e3, 4.0, 10),
        ("10^11 (z=10)", 1e11, 14.4e3, 3.5, 10),
    ]

    for name, Mvir, Rvir_pc, c, z in cases:
        g_vir_over_G = Mvir / Rvir_pc**2
        mc = mu_nfw(c)
        g_max_over_G = g_vir_over_G * c**2 / (2.0 * mc)
        above = "ABOVE" if g_max_over_G > G_CRIT_OVER_G else "below"
        print(f"  {name:25s}: g_vir/G = {g_vir_over_G:.1f} M_sun/pc^2, c = {c:.1f}, "
              f"g_max/G = {g_max_over_G:.1f} M_sun/pc^2  [{above} g_crit/G = {G_CRIT_OVER_G:.0f}]")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("BK25 FFB Validation Script")
    print("=" * 70)

    # Always run theory / sanity checks
    print_sanity_checks()

    print("\n--- Plot 1: NFW acceleration profiles (BK25 Fig 1 left) ---")
    plot_nfw_acceleration_profiles()

    print("\n--- Plot 2: g_max/G vs M_vir theory curves (Duffy) ---")
    plot_gmax_vs_mass_theory()

    print("\n--- Plot 3: g_max/G vs M_star theory curves (Duffy) ---")
    plot_gmax_vs_stellar_mass_theory()

    # Run diagnostics on actual SAGE output if available
    files = find_model_files(MODEL_DIR)
    if files:
        print(f"\n--- Loading SAGE output from {MODEL_DIR} ---")
        print(f"    Found {len(files)} model file(s)")
        header = read_header(MODEL_DIR)

        print("\n--- Plot 4-5: g_max/G vs Mvir & M_star (Vmax/Vvir from data) ---")
        plot_gmax_vs_mass_theory_vmax_vvir(header)

        plot_ffb_diagnostics(header)

        print("\n--- Plot: Mvir vs z by concentration ---")
        plot_mvir_z_by_concentration(header)
    else:
        print(f"\n  No model files found in {MODEL_DIR} — skipping SAGE diagnostics.")
        print("  Run SAGE first, then re-run this script.")

    print("\n" + "=" * 70)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("=" * 70)

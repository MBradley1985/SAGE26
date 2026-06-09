#!/usr/bin/env python
"""
Validate halo concentrations and g_max from SAGE26 BK25 FFB output.

Compares the code's lookup-table concentrations against Colossus,
checks g_max computation, and shows FFB activation statistics.

Usage:
    python plotting/validate_concentrations.py
    mpirun -np 4 python plotting/validate_concentrations.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py as h5

# MPI (optional — falls back to serial)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

# Optional: try to load a nice style
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_FILE = os.path.join(SCRIPT_DIR, 'kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_bk25')
CONC1_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_conc1')
CONC2_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_conc2')
FFB_CUT_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_cut')
BK25_CONC_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_bk25_conc')
NOSIGMOID_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_nosigmoid')
BK25_SMOOTH_MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_bk25_smooth')
LI24_MODEL = os.path.join(PROJECT_DIR, 'output', 'millennium')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'plots')
if rank == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical / code constants
G_NEWTON = 6.6743e-11       # m^3 kg^-1 s^-2
M_SUN_KG = 1.98892e30       # kg
PC_M     = 3.08568e16       # m
G_CONV   = G_NEWTON * M_SUN_KG / PC_M**2   # m/s^2 per (M_sun/pc^2)
G_CRIT_OVER_G = 3100.0      # M_sun / pc^2  (BK25 Table 1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def find_model_files(directory):
    pattern = os.path.join(directory, 'model_*.hdf5')
    # Exclude the combined file if present
    files = sorted([f for f in glob.glob(pattern) if 'model.hdf5' not in os.path.basename(f) or '_' in os.path.basename(f)])
    return files


def read_header(directory):
    files = find_model_files(directory)
    if not files:
        raise FileNotFoundError(f"No model_*.hdf5 files in {directory}")
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        header = {
            'hubble_h': float(sim.attrs['hubble_h']),
            'omega_matter': float(sim.attrs['omega_matter']),
            'omega_lambda': float(sim.attrs['omega_lambda']),
            'box_size': float(sim.attrs['box_size']),
            'redshifts': np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps': np.array(f['Header/output_snapshots'][:]),
        }
    # Sum frac_volume_processed across all MPI files
    total_fvp = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            if 'Header/Runtime' in f and 'frac_volume_processed' in f['Header/Runtime'].attrs:
                total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    header['volume_fraction'] = total_fvp if total_fvp > 0 else 1.0
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
# BK25 physics (Python reference implementation)
# ---------------------------------------------------------------------------
def mu_nfw(x):
    return np.log(1.0 + x) - x / (1.0 + x)


def g_max_over_G_from_Mvir_Rvir_c(Mvir_Msun, Rvir_pc, c):
    """g_max/G in Msun/pc^2, purely physical units."""
    g_vir_over_G = Mvir_Msun / Rvir_pc**2
    return g_vir_over_G * c**2 / (2.0 * mu_nfw(c))


# ---------------------------------------------------------------------------
# Colossus reference (if available)
# ---------------------------------------------------------------------------
def get_colossus_concentrations(logM_arr, z, h, Om, OL):
    """Return Ishiyama+21 200c concentrations from Colossus."""
    try:
        from colossus.cosmology import cosmology
        from colossus.halo import concentration
        params = {
            'flat': True, 'H0': h * 100, 'Om0': Om,
            'Ob0': 0.045, 'sigma8': 0.9, 'ns': 1.0, 'relspecies': False
        }
        cosmology.setCosmology('millennium_val', **params)
        masses = 10**logM_arr  # Msun/h
        c_vals = concentration.concentration(masses, '200c', z, model='ishiyama21')
        return c_vals
    except ImportError:
        print("  [Colossus not available — skipping direct comparison]")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    header = read_header(MODEL_DIR)
    h = header['hubble_h']
    Om = header['omega_matter']
    OL = header['omega_lambda']
    redshifts = header['redshifts']
    output_snaps = header['output_snaps']

    if rank == 0:
        print("=" * 70)
        print("SAGE26 Concentration & g_max Validation")
        print("=" * 70)
        print(f"Cosmology: h={h}, Omega_m={Om}, Omega_L={OL}")
        print(f"N output snapshots: {len(output_snaps)}")
        if size > 1:
            print(f"Running with {size} MPI ranks")
        print()

    props = ['Mvir', 'Rvir', 'Concentration', 'g_max', 'FFBRegime', 'Type',
             'StellarMass', 'ColdGas']
    li_props = ['Mvir', 'FFBRegime', 'Type', 'StellarMass', 'ColdGas']

    # Select snapshots spanning a range of redshifts
    snap_targets = [(63, 0.0), (50, 0.4), (40, 1.1), (30, 2.4),
                    (25, 3.5), (20, 5.3), (15, 8.2), (12, 10.1), (10, 11.9)]
    snap_list = []
    for sn, z_approx in snap_targets:
        if sn in output_snaps:
            snap_list.append(sn)
    snap_set = set(snap_list)

    # LI24_MODEL = os.path.join(PROJECT_DIR, 'output', 'millennium')

    # Code-unit constants (needed for threshold lines later)
    Hubble_code = 100.0
    G_code = 43.0071
    Msun_code = 1.989e33 / 1.989e43
    pc_code = 3.08568e18 / 3.08568e24
    g_crit_code = G_code * 3100.0 * Msun_code / pc_code**2 / h

    # ==================================================================
    # Distributed reading phase — each rank reads a subset of snapshots
    # ==================================================================
    my_snaps = [output_snaps[i] for i in range(len(output_snaps))
                if i % size == rank]

    # --- BK25 model ---
    my_all_data = {}
    my_fig6_z, my_fig6_logM, my_fig6_c, my_fig6_ffb = [], [], [], []

    for snap in my_snaps:
        z = redshifts[snap]
        data = read_snap(MODEL_DIR, snap, props)
        if 'Mvir' not in data or len(data['Mvir']) == 0:
            continue

        # Figure 6 data (all galaxies)
        c_arr = data['Concentration']
        mask_all = (data['Mvir'] > 0) & (c_arr > 0)
        logM_all = np.log10(np.clip(data['Mvir'][mask_all] * 1e10, 1e6, None))
        my_fig6_z.append(np.full(mask_all.sum(), z))
        my_fig6_logM.append(logM_all)
        my_fig6_c.append(c_arr[mask_all])
        my_fig6_ffb.append(data['FFBRegime'][mask_all])

        # all_data (centrals only, target snapshots)
        if snap in snap_set:
            Mvir_Msun = data['Mvir'] * 1e10 / h
            logM = np.log10(np.clip(Mvir_Msun * h, 1e6, None))
            Rvir_pc = data['Rvir'] / h * 1e6
            c = data['Concentration']
            gtype = data['Type']
            mask = (gtype == 0) & (data['Mvir'] > 0) & (c > 0)
            my_all_data[snap] = {
                'z': z,
                'logM': logM[mask],
                'Mvir_Msun': Mvir_Msun[mask],
                'Rvir_pc': Rvir_pc[mask],
                'c': c[mask],
                'g_max_code': data['g_max'][mask],
                'ffb': data['FFBRegime'][mask],
                'n_total': len(data['Mvir']),
                'n_centrals': int(mask.sum()),
            }

    # --- Li+ model ---
    my_li_fig3 = {}   # snap -> (z, n_cen, n_ffb)
    my_li_stats = {}  # snap -> dict with logM, ffb, smass
    my_li_fig7_z, my_li_fig7_logM, my_li_fig7_ffb = [], [], []

    for snap in my_snaps:
        z = redshifts[snap]
        li_data = read_snap(LI24_MODEL, snap, li_props)
        if 'Mvir' not in li_data or len(li_data['Mvir']) == 0:
            continue

        mask = li_data['Mvir'] > 0
        logM_tmp = np.log10(np.clip(li_data['Mvir'][mask] * 1e10, 1e6, None))
        ffb_tmp = li_data['FFBRegime'][mask]

        # Figure 7 data (all galaxies)
        my_li_fig7_z.append(np.full(mask.sum(), z))
        my_li_fig7_logM.append(logM_tmp)
        my_li_fig7_ffb.append(ffb_tmp)

        if snap in snap_set:
            # Figure 3 (centrals)
            li_cen = (li_data['Type'] == 0) & (li_data['Mvir'] > 0)
            n_cen = int(li_cen.sum())
            n_ffb = int((li_data['FFBRegime'][li_cen] == 1).sum())
            my_li_fig3[snap] = (z, n_cen, n_ffb)

            # Stats (all types, Mvir > 0)
            smass = li_data.get('StellarMass',
                                np.zeros_like(li_data['Mvir']))[mask]
            my_li_stats[snap] = {
                'z': z,
                'logM': logM_tmp,
                'ffb': ffb_tmp == 1,
                'smass': smass,
            }

    # --- ConcentrationOn=1 model (millennium_conc1) ---
    conc1_props = ['Mvir', 'Rvir', 'Concentration', 'FFBRegime', 'Type',
                   'StellarMass', 'ColdGas']
    my_conc1_data = {}
    my_conc1_fig6_z, my_conc1_fig6_logM, my_conc1_fig6_c = [], [], []
    for snap in my_snaps:
        z = redshifts[snap]
        data = read_snap(CONC1_MODEL_DIR, snap, conc1_props)
        if 'Mvir' not in data or len(data['Mvir']) == 0:
            continue
        logM = np.log10(np.clip(data['Mvir'] * 1e10, 1e6, None))
        c = data['Concentration']
        gtype = data['Type']

        # All-snapshot scatter data for fig6-style plot
        mask_all = (data['Mvir'] > 0) & (c > 0)
        my_conc1_fig6_z.append(np.full(mask_all.sum(), z))
        my_conc1_fig6_logM.append(logM[mask_all])
        my_conc1_fig6_c.append(c[mask_all])

        # Target-snapshot centrals data
        if snap in snap_set:
            mask = (gtype == 0) & (data['Mvir'] > 0) & (c > 0)
            my_conc1_data[snap] = {
                'z': z,
                'logM': logM[mask],
                'c': c[mask],
                'ffb': data['FFBRegime'][mask],
                'n_centrals': int(mask.sum()),
            }

    # --- ConcentrationOn=2 model (millennium_conc2) ---
    my_conc2_data = {}
    for snap in my_snaps:
        if snap not in snap_set:
            continue
        z = redshifts[snap]
        data = read_snap(CONC2_MODEL_DIR, snap, conc1_props)
        if 'Mvir' not in data or len(data['Mvir']) == 0:
            continue
        logM = np.log10(np.clip(data['Mvir'] * 1e10, 1e6, None))
        c = data['Concentration']
        gtype = data['Type']
        mask = (gtype == 0) & (data['Mvir'] > 0) & (c > 0)
        my_conc2_data[snap] = {
            'z': z,
            'logM': logM[mask],
            'c': c[mask],
            'ffb': data['FFBRegime'][mask],
            'n_centrals': int(mask.sum()),
        }

    # --- Li+24 with sigmoid cutoff model (millennium_ffb_cut) ---
    my_cut_fig7_z, my_cut_fig7_logM, my_cut_fig7_ffb = [], [], []
    for snap in my_snaps:
        z = redshifts[snap]
        cut_data = read_snap(FFB_CUT_MODEL_DIR, snap, li_props)
        if 'Mvir' not in cut_data or len(cut_data['Mvir']) == 0:
            continue
        mask = cut_data['Mvir'] > 0
        logM_tmp = np.log10(np.clip(cut_data['Mvir'][mask] * 1e10, 1e6, None))
        ffb_tmp = cut_data['FFBRegime'][mask]
        my_cut_fig7_z.append(np.full(mask.sum(), z))
        my_cut_fig7_logM.append(logM_tmp)
        my_cut_fig7_ffb.append(ffb_tmp)

    # --- BK25 with Vmax/Vvir concentration (millennium_ffb_bk25_conc) ---
    my_bk25conc_fig7_z, my_bk25conc_fig7_logM, my_bk25conc_fig7_ffb = [], [], []
    for snap in my_snaps:
        z = redshifts[snap]
        bc_data = read_snap(BK25_CONC_MODEL_DIR, snap, li_props)
        if 'Mvir' not in bc_data or len(bc_data['Mvir']) == 0:
            continue
        mask = bc_data['Mvir'] > 0
        logM_tmp = np.log10(np.clip(bc_data['Mvir'][mask] * 1e10, 1e6, None))
        ffb_tmp = bc_data['FFBRegime'][mask]
        my_bk25conc_fig7_z.append(np.full(mask.sum(), z))
        my_bk25conc_fig7_logM.append(logM_tmp)
        my_bk25conc_fig7_ffb.append(ffb_tmp)

    # --- Li+24 no sigmoid (millennium_nosigmoid) ---
    my_nosig_fig7_z, my_nosig_fig7_logM, my_nosig_fig7_ffb = [], [], []
    for snap in my_snaps:
        z = redshifts[snap]
        ns_data = read_snap(NOSIGMOID_MODEL_DIR, snap, li_props)
        if 'Mvir' not in ns_data or len(ns_data['Mvir']) == 0:
            continue
        mask = ns_data['Mvir'] > 0
        logM_tmp = np.log10(np.clip(ns_data['Mvir'][mask] * 1e10, 1e6, None))
        ffb_tmp = ns_data['FFBRegime'][mask]
        my_nosig_fig7_z.append(np.full(mask.sum(), z))
        my_nosig_fig7_logM.append(logM_tmp)
        my_nosig_fig7_ffb.append(ffb_tmp)

    # --- BK25 with log-normal concentration scatter (millennium_ffb_bk25_smooth) ---
    my_bk25smooth_fig7_z, my_bk25smooth_fig7_logM, my_bk25smooth_fig7_ffb = [], [], []
    for snap in my_snaps:
        z = redshifts[snap]
        bs_data = read_snap(BK25_SMOOTH_MODEL_DIR, snap, li_props)
        if 'Mvir' not in bs_data or len(bs_data['Mvir']) == 0:
            continue
        mask = bs_data['Mvir'] > 0
        logM_tmp = np.log10(np.clip(bs_data['Mvir'][mask] * 1e10, 1e6, None))
        ffb_tmp = bs_data['FFBRegime'][mask]
        my_bk25smooth_fig7_z.append(np.full(mask.sum(), z))
        my_bk25smooth_fig7_logM.append(logM_tmp)
        my_bk25smooth_fig7_ffb.append(ffb_tmp)

    # ==================================================================
    # Gather to rank 0
    # ==================================================================
    def concat_or_empty(arrays):
        return np.concatenate(arrays) if arrays else np.array([])

    if comm is not None:
        all_data_parts = comm.gather(my_all_data, root=0)
        fig6_z_parts = comm.gather(concat_or_empty(my_fig6_z), root=0)
        fig6_logM_parts = comm.gather(concat_or_empty(my_fig6_logM), root=0)
        fig6_c_parts = comm.gather(concat_or_empty(my_fig6_c), root=0)
        fig6_ffb_parts = comm.gather(concat_or_empty(my_fig6_ffb), root=0)
        li_fig3_parts = comm.gather(my_li_fig3, root=0)
        li_stats_parts = comm.gather(my_li_stats, root=0)
        li_fig7_z_parts = comm.gather(concat_or_empty(my_li_fig7_z), root=0)
        li_fig7_logM_parts = comm.gather(concat_or_empty(my_li_fig7_logM), root=0)
        li_fig7_ffb_parts = comm.gather(concat_or_empty(my_li_fig7_ffb), root=0)
        conc1_data_parts = comm.gather(my_conc1_data, root=0)
        conc1_fig6_z_parts = comm.gather(concat_or_empty(my_conc1_fig6_z), root=0)
        conc1_fig6_logM_parts = comm.gather(concat_or_empty(my_conc1_fig6_logM), root=0)
        conc1_fig6_c_parts = comm.gather(concat_or_empty(my_conc1_fig6_c), root=0)
        conc2_data_parts = comm.gather(my_conc2_data, root=0)
        cut_fig7_z_parts = comm.gather(concat_or_empty(my_cut_fig7_z), root=0)
        cut_fig7_logM_parts = comm.gather(concat_or_empty(my_cut_fig7_logM), root=0)
        cut_fig7_ffb_parts = comm.gather(concat_or_empty(my_cut_fig7_ffb), root=0)
        bk25conc_fig7_z_parts = comm.gather(concat_or_empty(my_bk25conc_fig7_z), root=0)
        bk25conc_fig7_logM_parts = comm.gather(concat_or_empty(my_bk25conc_fig7_logM), root=0)
        bk25conc_fig7_ffb_parts = comm.gather(concat_or_empty(my_bk25conc_fig7_ffb), root=0)
        nosig_fig7_z_parts = comm.gather(concat_or_empty(my_nosig_fig7_z), root=0)
        nosig_fig7_logM_parts = comm.gather(concat_or_empty(my_nosig_fig7_logM), root=0)
        nosig_fig7_ffb_parts = comm.gather(concat_or_empty(my_nosig_fig7_ffb), root=0)
        bk25smooth_fig7_z_parts = comm.gather(concat_or_empty(my_bk25smooth_fig7_z), root=0)
        bk25smooth_fig7_logM_parts = comm.gather(concat_or_empty(my_bk25smooth_fig7_logM), root=0)
        bk25smooth_fig7_ffb_parts = comm.gather(concat_or_empty(my_bk25smooth_fig7_ffb), root=0)

        if rank == 0:
            all_data = {}
            for d in all_data_parts:
                all_data.update(d)
            fig6_z = np.concatenate([p for p in fig6_z_parts if len(p) > 0])
            fig6_logM = np.concatenate([p for p in fig6_logM_parts if len(p) > 0])
            fig6_c = np.concatenate([p for p in fig6_c_parts if len(p) > 0])
            fig6_ffb = np.concatenate([p for p in fig6_ffb_parts if len(p) > 0])
            li_fig3 = {}
            for d in li_fig3_parts:
                li_fig3.update(d)
            li_stats = {}
            for d in li_stats_parts:
                li_stats.update(d)
            li_z_all = np.concatenate([p for p in li_fig7_z_parts if len(p) > 0])
            li_logM_all = np.concatenate([p for p in li_fig7_logM_parts if len(p) > 0])
            li_ffb_all = np.concatenate([p for p in li_fig7_ffb_parts if len(p) > 0])
            conc1_data = {}
            for d in conc1_data_parts:
                conc1_data.update(d)
            conc1_fig6_z = np.concatenate([p for p in conc1_fig6_z_parts if len(p) > 0])
            conc1_fig6_logM = np.concatenate([p for p in conc1_fig6_logM_parts if len(p) > 0])
            conc1_fig6_c = np.concatenate([p for p in conc1_fig6_c_parts if len(p) > 0])
            conc2_data = {}
            for d in conc2_data_parts:
                conc2_data.update(d)
            cut_z_all = concat_or_empty([p for p in cut_fig7_z_parts if len(p) > 0])
            cut_logM_all = concat_or_empty([p for p in cut_fig7_logM_parts if len(p) > 0])
            cut_ffb_all = concat_or_empty([p for p in cut_fig7_ffb_parts if len(p) > 0])
            bk25conc_z_all = concat_or_empty([p for p in bk25conc_fig7_z_parts if len(p) > 0])
            bk25conc_logM_all = concat_or_empty([p for p in bk25conc_fig7_logM_parts if len(p) > 0])
            bk25conc_ffb_all = concat_or_empty([p for p in bk25conc_fig7_ffb_parts if len(p) > 0])
            nosig_z_all = concat_or_empty([p for p in nosig_fig7_z_parts if len(p) > 0])
            nosig_logM_all = concat_or_empty([p for p in nosig_fig7_logM_parts if len(p) > 0])
            nosig_ffb_all = concat_or_empty([p for p in nosig_fig7_ffb_parts if len(p) > 0])
            bk25smooth_z_all = concat_or_empty([p for p in bk25smooth_fig7_z_parts if len(p) > 0])
            bk25smooth_logM_all = concat_or_empty([p for p in bk25smooth_fig7_logM_parts if len(p) > 0])
            bk25smooth_ffb_all = concat_or_empty([p for p in bk25smooth_fig7_ffb_parts if len(p) > 0])
    else:
        all_data = my_all_data
        fig6_z = concat_or_empty(my_fig6_z)
        fig6_logM = concat_or_empty(my_fig6_logM)
        fig6_c = concat_or_empty(my_fig6_c)
        fig6_ffb = concat_or_empty(my_fig6_ffb)
        li_fig3 = my_li_fig3
        li_stats = my_li_stats
        li_z_all = concat_or_empty(my_li_fig7_z)
        li_logM_all = concat_or_empty(my_li_fig7_logM)
        li_ffb_all = concat_or_empty(my_li_fig7_ffb)
        conc1_data = my_conc1_data
        conc1_fig6_z = concat_or_empty(my_conc1_fig6_z)
        conc1_fig6_logM = concat_or_empty(my_conc1_fig6_logM)
        conc1_fig6_c = concat_or_empty(my_conc1_fig6_c)
        conc2_data = my_conc2_data
        cut_z_all = concat_or_empty(my_cut_fig7_z)
        cut_logM_all = concat_or_empty(my_cut_fig7_logM)
        cut_ffb_all = concat_or_empty(my_cut_fig7_ffb)
        bk25conc_z_all = concat_or_empty(my_bk25conc_fig7_z)
        bk25conc_logM_all = concat_or_empty(my_bk25conc_fig7_logM)
        bk25conc_ffb_all = concat_or_empty(my_bk25conc_fig7_ffb)
        nosig_z_all = concat_or_empty(my_nosig_fig7_z)
        nosig_logM_all = concat_or_empty(my_nosig_fig7_logM)
        nosig_ffb_all = concat_or_empty(my_nosig_fig7_ffb)
        bk25smooth_z_all = concat_or_empty(my_bk25smooth_fig7_z)
        bk25smooth_logM_all = concat_or_empty(my_bk25smooth_fig7_logM)
        bk25smooth_ffb_all = concat_or_empty(my_bk25smooth_fig7_ffb)

    # ==================================================================
    # Non-root ranks are done
    # ==================================================================
    if rank != 0:
        return

    # ==================================================================
    # Print summary statistics
    # ==================================================================
    print(f"{'Snap':>4s} {'z':>6s} {'N_cen':>7s} {'c_med':>7s} {'c_16':>6s} "
          f"{'c_84':>6s} {'N_FFB':>6s} {'f_FFB':>7s} {'logM_FFB_min':>12s}")
    print("-" * 75)
    for snap in sorted(all_data.keys(), reverse=True):
        d = all_data[snap]
        z = d['z']
        c_arr = d['c']
        ffb = d['ffb']
        n_ffb = int((ffb == 1).sum())
        f_ffb = n_ffb / d['n_centrals'] if d['n_centrals'] > 0 else 0
        c_med = np.median(c_arr)
        c_16 = np.percentile(c_arr, 16)
        c_84 = np.percentile(c_arr, 84)
        logM_ffb = d['logM'][ffb == 1]
        logM_ffb_min = f"{logM_ffb.min():.2f}" if len(logM_ffb) > 0 else "—"
        print(f"{snap:>4d} {z:>6.2f} {d['n_centrals']:>7d} {c_med:>7.2f} {c_16:>6.2f} "
              f"{c_84:>6.2f} {n_ffb:>6d} {f_ffb:>7.3f} {logM_ffb_min:>12s}")
    print()

    # ======================================================================
    # Figure 1: Concentration vs Mass at multiple redshifts (3 columns)
    #   Col 0: ConcentrationOn=1 — Ishiyama+21 lookup table
    #   Col 1: ConcentrationOn=2 — Vmax/Vvir from simulation
    #   Col 2: ConcentrationOn=3 — Vmax/Vvir + infall freeze for satellites
    # ======================================================================
    sorted_snaps = sorted(all_data.keys(), reverse=True)[:9]
    n_rows = len(sorted_snaps)
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 4 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_data_sources = [conc1_data, conc2_data, all_data]
    col_labels = [
        'Ishiyama+21 lookup table',
        r'$V_{\rm max}/V_{\rm vir}$ from simulation',
        r'$V_{\rm max}/V_{\rm vir}$ + infall freeze',
    ]

    logM_bins = np.arange(8, 16.1, 0.3)
    bin_centers = 0.5 * (logM_bins[:-1] + logM_bins[1:])

    for idx, snap in enumerate(sorted_snaps):
        z = all_data[snap]['z'] if snap in all_data else redshifts[snap]

        # Compute Colossus reference once per row
        ref_d = all_data.get(snap, conc1_data.get(snap, conc2_data.get(snap)))
        logM_grid = None
        c_ref = None
        if ref_d is not None and len(ref_d['logM']) > 0:
            logM_grid = np.linspace(max(ref_d['logM'].min(), 8.0),
                                    min(ref_d['logM'].max(), 16.0), 100)
            c_ref = get_colossus_concentrations(logM_grid, z, h, Om, OL)

        for col_idx, src in enumerate(col_data_sources):
            ax = axes[idx, col_idx]

            if snap in src and len(src[snap]['logM']) > 0:
                d = src[snap]
                normal = d['ffb'] == 0
                ffb_mask = d['ffb'] == 1
                ax.scatter(d['logM'][normal], d['c'][normal], s=1, alpha=0.15,
                           color='C0', rasterized=True, label='Normal')
                if ffb_mask.any():
                    ax.scatter(d['logM'][ffb_mask], d['c'][ffb_mask], s=4, alpha=0.5,
                               color='C3', rasterized=True, label='FFB')

                if c_ref is not None:
                    ax.plot(logM_grid, c_ref, 'k-', lw=2, label='Ishiyama+21')

                bin_idx_arr = np.digitize(d['logM'], logM_bins)
                c_median = np.array([np.median(d['c'][bin_idx_arr == i])
                                     if (bin_idx_arr == i).sum() > 5 else np.nan
                                     for i in range(1, len(logM_bins))])
                valid = ~np.isnan(c_median)
                ax.plot(bin_centers[valid], c_median[valid], 'o-', color='C1',
                        ms=4, lw=1.5, label='SAGE median')
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='0.5')

            ax.set_ylim(0.5, 30)
            ax.set_yscale('log')
            ax.set_xlim(8, 15)
            ax.grid(False)
            ax.text(0.95, 0.95, f'$z = {z:.1f}$', transform=ax.transAxes,
                    ha='right', va='top', fontsize=12)
            if idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='lower left')

    # Column headers
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].text(0.5, 1.15, label,
                              transform=axes[0, col_idx].transAxes,
                              ha='center', fontsize=13, fontweight='bold')

    fig.supxlabel(r'$\log_{10}(M_{200c}\ /\ M_\odot\,h^{-1})$', fontsize=14)
    fig.supylabel('Concentration $c$', fontsize=14)
    fig.suptitle('SAGE26 Concentration–Mass Relation vs Ishiyama+21 (200c)', fontsize=15, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'concentration_vs_mass.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 2: g_max verification — code vs Python recomputation
    # ======================================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    # Show all snapshots (up to 9) to cover full redshift range to z~12
    check_snaps = sorted(all_data.keys(), reverse=True)[:9]
    for idx, snap in enumerate(check_snaps):
        ax = axes[idx]
        d = all_data[snap]
        z = d['z']

        # Recompute g_max/G in physical units from Mvir, Rvir, c
        g_vir_over_G = d['Mvir_Msun'] / d['Rvir_pc']**2  # Msun/pc^2
        g_max_over_G_py = g_vir_over_G * d['c']**2 / (2.0 * mu_nfw(d['c']))

        ratio_py = g_max_over_G_py / G_CRIT_OVER_G

        # Plot g_max/G vs mass
        ax.scatter(d['logM'], g_max_over_G_py, s=1, alpha=0.15, color='C0', rasterized=True)
        ax.axhline(G_CRIT_OVER_G, color='red', ls='--', lw=2, label=r'$g_{\rm crit}/G = 3100$')

        # Mark FFB galaxies
        ffb_mask = d['ffb'] == 1
        if ffb_mask.any():
            ax.scatter(d['logM'][ffb_mask], g_max_over_G_py[ffb_mask],
                       s=6, alpha=0.5, color='C3', rasterized=True, zorder=5)

        ax.set_yscale('log')
        ax.set_xlim(8, 15)
        ax.set_ylim(1, 1e6)
        ax.set_title(f'z = {z:.1f}', fontsize=12)
        ax.set_xlabel(r'$\log_{10}(M_{200c}\ /\ M_\odot\,h^{-1})$')
        ax.set_ylabel(r'$g_{\rm max}/G\ [M_\odot\,{\rm pc}^{-2}]$')
        ax.grid(False)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(r'$g_{\rm max}/G$ vs Halo Mass — BK25 FFB Threshold', fontsize=15, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'gmax_vs_mass.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 3: FFB fraction vs redshift, and threshold mass vs redshift
    # ======================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    z_arr = []
    f_ffb_arr = []
    logM_thresh_arr = []
    c_median_arr = []

    for snap in sorted(all_data.keys()):
        d = all_data[snap]
        z = d['z']
        n_cen = d['n_centrals']
        if n_cen == 0:
            continue
        n_ffb = (d['ffb'] == 1).sum()
        z_arr.append(z)
        f_ffb_arr.append(n_ffb / n_cen)
        c_median_arr.append(np.median(d['c']))

        # Threshold mass: minimum mass in FFB
        logM_ffb = d['logM'][d['ffb'] == 1]
        if len(logM_ffb) > 0:
            logM_thresh_arr.append((z, np.percentile(logM_ffb, 5)))
        else:
            logM_thresh_arr.append((z, np.nan))

    z_arr = np.array(z_arr)
    f_ffb_arr = np.array(f_ffb_arr)
    c_median_arr = np.array(c_median_arr)

    # Panel 1: FFB fraction vs z
    ax1.plot(z_arr, f_ffb_arr, 'o-', color='C3', lw=2, ms=6, label='BK25')

    # Li+24 FFB fraction from pre-read data
    li24_z_arr = []
    li24_f_ffb = []
    for snap in sorted(all_data.keys()):
        if snap not in li_fig3:
            continue
        z, n_cen, n_ffb = li_fig3[snap]
        if n_cen == 0:
            continue
        li24_z_arr.append(z)
        li24_f_ffb.append(n_ffb / n_cen)
    ax1.plot(li24_z_arr, li24_f_ffb, 's--', color='C0', lw=2, ms=5, label='Li+24')

    # Li+24 with sigmoid cutoff FFB fraction
    cut_z_arr_fig3 = []
    cut_f_ffb_fig3 = []
    for snap in sorted(all_data.keys()):
        z = redshifts[snap]
        cut_data_snap = read_snap(FFB_CUT_MODEL_DIR, snap, li_props)
        if 'Mvir' not in cut_data_snap or len(cut_data_snap['Mvir']) == 0:
            continue
        cut_cen = (cut_data_snap['Type'] == 0) & (cut_data_snap['Mvir'] > 0)
        n_cen_cut = int(cut_cen.sum())
        if n_cen_cut == 0:
            continue
        n_ffb_cut = int((cut_data_snap['FFBRegime'][cut_cen] == 1).sum())
        cut_z_arr_fig3.append(z)
        cut_f_ffb_fig3.append(n_ffb_cut / n_cen_cut)
    ax1.plot(cut_z_arr_fig3, cut_f_ffb_fig3, 'D-.', color='C2', lw=2, ms=5,
             label='Li+24 (with cutoff)')

    ax1.set_xlabel('Redshift $z$', fontsize=13)
    ax1.set_ylabel('FFB Fraction (centrals)', fontsize=13)
    ax1.set_title('FFB Activation vs Redshift', fontsize=13)
    ax1.set_xlim(-0.5, max(z_arr) + 0.5)
    all_f_ffb = list(f_ffb_arr) + li24_f_ffb + cut_f_ffb_fig3
    ymax = max(all_f_ffb) if all_f_ffb else 0.1
    ax1.set_ylim(-0.01, ymax * 1.2 if ymax > 0 else 0.1)
    ax1.legend(fontsize=9)
    ax1.grid(False)

    # Panel 2: FFB threshold mass vs z
    z_thresh = np.array([t[0] for t in logM_thresh_arr])
    logM_thresh = np.array([t[1] for t in logM_thresh_arr])
    valid = ~np.isnan(logM_thresh)
    ax2.plot(z_thresh[valid], logM_thresh[valid], 's-', color='C0', lw=2, ms=6,
             label='SAGE (5th %ile FFB mass)')

    # Compute the actual g_max = g_crit threshold for SAGE's own cosmology/mass def
    z_theory = np.linspace(4, 15, 50)
    logM_theory = np.full_like(z_theory, np.nan)
    try:
        from colossus.cosmology import cosmology as colossus_cosmo
        from colossus.halo import concentration as colossus_conc
        from scipy.optimize import brentq
        colossus_cosmo.setCosmology('mill_thresh', flat=True, H0=h*100,
                                     Om0=Om, Ob0=0.045, sigma8=0.9, ns=1.0,
                                     relspecies=False)

        def g_max_minus_g_crit(logM_Msun_h, z_val):
            M = 10**logM_Msun_h  # Msun/h
            c = colossus_conc.concentration(M, '200c', z_val, model='ishiyama21')
            Mvir_code = M / 1e10  # code mass (10^10 Msun/h)
            zp1 = 1.0 + z_val
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac = 1.0 / (200.0 * 4 * np.pi / 3.0 * rhocrit)
            Rvir_code = (Mvir_code * fac)**(1./3.)
            g_vir = G_code * Mvir_code / Rvir_code**2
            mu_c = np.log(1 + c) - c / (1 + c)
            g_max = g_vir * c**2 / (2 * mu_c)
            return g_max - g_crit_code

        for i, zv in enumerate(z_theory):
            try:
                logM_sol = brentq(g_max_minus_g_crit, 8.0, 15.0, args=(zv,))
                logM_theory[i] = logM_sol
            except ValueError:
                pass

        theory_valid = ~np.isnan(logM_theory)
        ax2.plot(z_theory[theory_valid], logM_theory[theory_valid], 'k-', lw=2,
                 label=r'$g_{\rm max} = g_{\rm crit}$ (Millennium, 200c)')
    except ImportError:
        pass

    # BK25 prediction (Planck cosmology, virial def) — converted to Msun/h
    z_bk = np.linspace(4, 15, 50)
    logM_bk = 10.8 + np.log10(h) - 6.0 * np.log10((1 + z_bk) / (1 + 10))
    ax2.plot(z_bk, logM_bk, 'k--', lw=1.5, alpha=0.5,
             label=r'BK25 (Planck, vir): $M \propto (1+z)^{-6}$')

    # Li+24 threshold mass — converted to Msun/h
    z_li = np.linspace(0, 15, 200)
    logM_li = 10.8 + np.log10(h) - 6.2 * np.log10((1 + z_li) / 10.0)
    ax2.plot(z_li, logM_li, 'b--', lw=2,
             label=r'Li+24: $M \propto (1+z)^{-6.2}$')

    ax2.set_xlabel('Redshift $z$', fontsize=13)
    ax2.set_ylabel(r'$\log_{10}(M_{\rm thresh}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax2.set_title('FFB Threshold Mass vs Redshift', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(False)
    ax2.set_xlim(-0.5, max(z_arr) + 0.5)

    # Panel 3: median concentration vs z
    ax3.plot(z_arr, c_median_arr, 'o-', color='C2', lw=2, ms=6,
             label=r'$V_{\rm max}/V_{\rm vir}$ + infall freeze')

    # Add ConcentrationOn=1 (Ishiyama+21 lookup table) median concentration
    conc1_z_arr = []
    conc1_c_median_arr = []
    for snap in sorted(conc1_data.keys()):
        d2 = conc1_data[snap]
        if d2['n_centrals'] == 0:
            continue
        conc1_z_arr.append(d2['z'])
        conc1_c_median_arr.append(np.median(d2['c']))
    ax3.plot(conc1_z_arr, conc1_c_median_arr, 's--', color='C4', lw=2, ms=6,
             label='Ishiyama+21 lookup table')

    # Add ConcentrationOn=2 (Vmax/Vvir from simulation) median concentration
    conc2_z_arr = []
    conc2_c_median_arr = []
    for snap in sorted(conc2_data.keys()):
        d2 = conc2_data[snap]
        if d2['n_centrals'] == 0:
            continue
        conc2_z_arr.append(d2['z'])
        conc2_c_median_arr.append(np.median(d2['c']))
    ax3.plot(conc2_z_arr, conc2_c_median_arr, 'D-.', color='C5', lw=2, ms=6,
             label=r'$V_{\rm max}/V_{\rm vir}$')

    ax3.set_xlabel('Redshift $z$', fontsize=13)
    ax3.set_ylabel('Median Concentration (centrals)', fontsize=13)
    ax3.set_title('Median Concentration vs Redshift', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(False)
    ax3.set_xlim(-0.5, max(z_arr) + 0.5)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'ffb_summary.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 4: Residuals — SAGE concentration vs Colossus
    # ======================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    check_snaps = sorted(all_data.keys(), reverse=True)[:6]
    for idx, snap in enumerate(check_snaps):
        ax = axes[idx]
        d = all_data[snap]
        z = d['z']

        c_ref = get_colossus_concentrations(d['logM'], z, h, Om, OL)
        if c_ref is None:
            ax.text(0.5, 0.5, 'Colossus N/A', transform=ax.transAxes, ha='center')
            continue

        residual = (d['c'] - c_ref) / c_ref * 100  # percent

        # Bin by mass
        logM_bins = np.arange(8, 16.1, 0.3)
        bin_centers = 0.5 * (logM_bins[:-1] + logM_bins[1:])
        bin_idx = np.digitize(d['logM'], logM_bins)
        res_med = np.array([np.median(residual[bin_idx == i])
                            if (bin_idx == i).sum() > 5 else np.nan
                            for i in range(1, len(logM_bins))])
        res_16 = np.array([np.percentile(residual[bin_idx == i], 16)
                           if (bin_idx == i).sum() > 5 else np.nan
                           for i in range(1, len(logM_bins))])
        res_84 = np.array([np.percentile(residual[bin_idx == i], 84)
                           if (bin_idx == i).sum() > 5 else np.nan
                           for i in range(1, len(logM_bins))])

        valid = ~np.isnan(res_med)
        ax.fill_between(bin_centers[valid], res_16[valid], res_84[valid],
                        alpha=0.3, color='C0')
        ax.plot(bin_centers[valid], res_med[valid], 'o-', color='C0', ms=4, lw=1.5)
        ax.axhline(0, color='k', ls='-', lw=0.5)
        ax.set_title(f'z = {z:.1f}', fontsize=12)
        ax.set_xlabel(r'$\log_{10}(M_{200c}\ /\ M_\odot\,h^{-1})$')
        ax.set_ylabel('Residual [%]')
        ax.set_ylim(-5, 5)
        ax.set_xlim(8, 15)
        ax.grid(False)

    fig.suptitle('Concentration Residuals: (SAGE $-$ Colossus) / Colossus',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'concentration_residuals.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 5: g_max consistency check — code vs physical recomputation
    # ======================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, snap in enumerate(check_snaps):
        ax = axes[idx]
        d = all_data[snap]
        z = d['z']

        # Use lookup table concentration (matching what the code uses for g_max)
        c_lookup = get_colossus_concentrations(d['logM'], z, h, Om, OL)
        if c_lookup is None:
            ax.text(0.5, 0.5, 'Colossus N/A', transform=ax.transAxes, ha='center')
            continue
        c_lookup = np.clip(c_lookup, 1.0, None)

        # Physical g_max/G using lookup table c
        g_vir_over_G = d['Mvir_Msun'] / d['Rvir_pc']**2
        g_max_phys = g_vir_over_G * c_lookup**2 / (2.0 * mu_nfw(c_lookup))

        # Code g_max
        g_max_c = d['g_max_code']

        # These should be proportional. Find the proportionality constant.
        valid = (g_max_c > 0) & (g_max_phys > 0)
        if valid.sum() > 0:
            ratio = g_max_c[valid] / g_max_phys[valid]
            ratio_med = np.median(ratio)
            ratio_std = np.std(ratio) / ratio_med * 100  # percent scatter

            ax.scatter(np.log10(g_max_phys[valid]), ratio / ratio_med,
                       s=1, alpha=0.15, color='C0', rasterized=True)
            ax.axhline(1.0, color='k', ls='-', lw=1)
            ax.set_ylim(0.98, 1.02)
            ax.set_xlabel(r'$\log_{10}(g_{\rm max}/G)$ [physical]')
            ax.set_ylabel(r'Code / Physical (normalised)')
            ax.set_title(f'z = {z:.1f}  scatter = {ratio_std:.3f}%', fontsize=12)
            ax.grid(False)

    fig.suptitle(r'$g_{\rm max}$ Consistency: Code vs Physical Recomputation', fontsize=14, y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'gmax_consistency.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 6: Mvir vs Redshift coloured by concentration (3 panels)
    #   Panel 1: ConcentrationOn=1 (Ishiyama+21 lookup table) all galaxies
    #   Panel 2: ConcentrationOn=3 (Vmax/Vvir + infall freeze) all galaxies
    #   Panel 3: ConcentrationOn=3 FFB galaxies highlighted
    # ======================================================================
    all_c = np.concatenate([fig6_c, conc1_fig6_c])
    c_vmin, c_vmax = np.percentile(all_c, [2, 98])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    # --- Panel 1: ConcentrationOn=1 (lookup table) all galaxies ---
    ax0.scatter(conc1_fig6_z, conc1_fig6_logM, c=conc1_fig6_c, s=2, alpha=0.3,
                cmap='viridis', vmin=c_vmin, vmax=c_vmax, rasterized=True)
    ax0.set_xlabel('Redshift $z$', fontsize=13)
    ax0.set_ylabel(r'$\log_{10}(M_{200c}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax0.set_title('Ishiyama+21 Lookup Table', fontsize=13)
    ax0.set_ylim(10, 15)
    ax0.grid(False)

    # --- Panel 2: ConcentrationOn=3 (Vmax/Vvir) all galaxies ---
    ax1.scatter(fig6_z, fig6_logM, c=fig6_c, s=2, alpha=0.3,
                cmap='viridis', vmin=c_vmin, vmax=c_vmax, rasterized=True)
    ax1.set_xlabel('Redshift $z$', fontsize=13)
    ax1.set_title(r'$V_{\rm max}/V_{\rm vir}$ + Infall Freeze', fontsize=13)
    ax1.grid(False)

    # --- Panel 3: ConcentrationOn=3 FFB highlighted ---
    normal = fig6_ffb == 0
    ffb = fig6_ffb == 1

    ax2.scatter(fig6_z[normal], fig6_logM[normal], c='0.8', s=2, alpha=0.2,
                rasterized=True, label='Normal')
    sc_last = ax2.scatter(fig6_z[ffb], fig6_logM[ffb], c=fig6_c[ffb], s=60, alpha=0.9,
                          marker='*', cmap='viridis', vmin=c_vmin, vmax=c_vmax,
                          edgecolors='k', linewidths=0.3, rasterized=True, zorder=5,
                          label='FFB')

    # Overplot g_max = g_crit threshold lines for fixed concentrations
    z_line = np.linspace(0, 14, 200)
    c_fixed_vals = [3, 3.25, 3.5, 3.75, 4, 5, 7, 10]
    colors_c = plt.cm.plasma(np.linspace(0.15, 0.85, len(c_fixed_vals)))
    for c_fix, col in zip(c_fixed_vals, colors_c):
        mu_c = np.log(1 + c_fix) - c_fix / (1 + c_fix)
        logM_line = []
        for zv in z_line:
            zp1 = 1.0 + zv
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac200 = 200.0 * 4.0 * np.pi / 3.0 * rhocrit
            coeff = G_code * fac200**(2.0/3.0) * c_fix**2 / (2.0 * mu_c)
            M_code = (g_crit_code / coeff)**3
            M_Msun_h = M_code * 1e10  # Msun/h
            logM_line.append(np.log10(M_Msun_h))
        ax2.plot(z_line, logM_line, '-', color=col, lw=1.5, alpha=0.9,
                 label=f'$c = {c_fix:g}$')

    # --- Threshold lines ---
    z_th = np.linspace(0, 14, 200)

    # 1) Li+24 / SAGE25: log10(M/[Msun/h]) = 10.8 + log10(h) - 6.2*log10((1+z)/10)
    logM_li24 = 10.8 + np.log10(h) - 6.2 * np.log10((1 + z_th) / 10.0)
    ax2.plot(z_th, logM_li24, 'b--', lw=2, label=r'Li+24')

    # 2) BK25 (Planck, virial): log10(M/Msun) = 10.8 - 6*log10((1+z)/11), convert to Msun/h
    logM_bk25 = 10.8 + np.log10(h) - 6.0 * np.log10((1 + z_th) / 11.0)
    ax2.plot(z_th, logM_bk25, 'r--', lw=2, label=r'BK25 (Planck, vir)')

    # 3) SAGE26 BK25 (Millennium, 200c): root-find g_max = g_crit at each z
    logM_sage26 = np.full_like(z_th, np.nan)
    try:
        from colossus.cosmology import cosmology as colossus_cosmo
        from colossus.halo import concentration as colossus_conc
        from scipy.optimize import brentq
        colossus_cosmo.setCosmology('mill_fig6', flat=True, H0=h*100,
                                     Om0=Om, Ob0=0.045, sigma8=0.9, ns=1.0,
                                     relspecies=False)

        def g_max_minus_g_crit_fig6(logM_Msun_h, z_val):
            M = 10**logM_Msun_h
            c = colossus_conc.concentration(M, '200c', z_val, model='ishiyama21')
            Mvir_code = M / 1e10
            zp1 = 1.0 + z_val
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac = 1.0 / (200.0 * 4 * np.pi / 3.0 * rhocrit)
            Rvir_code = (Mvir_code * fac)**(1./3.)
            g_vir = G_code * Mvir_code / Rvir_code**2
            mu_c = np.log(1 + c) - c / (1 + c)
            g_max = g_vir * c**2 / (2 * mu_c)
            return g_max - g_crit_code

        for i, zv in enumerate(z_th):
            try:
                logM_sage26[i] = brentq(g_max_minus_g_crit_fig6, 8.0, 15.0, args=(zv,))
            except ValueError:
                pass
        valid_s26 = ~np.isnan(logM_sage26)
        ax2.plot(z_th[valid_s26], logM_sage26[valid_s26], 'r-', lw=2.5,
                 label=r'BK25 (Mill, 200c)')
    except ImportError:
        pass

    ax2.set_xlabel('Redshift $z$', fontsize=13)
    ax2.set_title('FFB Galaxies Highlighted', fontsize=13)
    ax2.legend(fontsize=8, loc='upper right', ncol=3)
    ax2.grid(False)

    # Single colorbar after the last panel
    cb = plt.colorbar(sc_last, ax=ax2, pad=0.02)
    cb.set_label('Concentration $c$', fontsize=12)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'mvir_vs_redshift_concentration.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Li+ vs g_max selection: per-snapshot comparison statistics
    # ======================================================================
    print()
    print("=" * 70)
    print("Li+ (Sigmoid) vs g_max Selection Comparison")
    print("=" * 70)

    # Li+24 threshold: logM_thresh(z) = 10.8 + log10(h) - 6.2*log10((1+z)/10)
    def li24_logM_thresh(z_val):
        return 10.8 + np.log10(h) - 6.2 * np.log10((1 + z_val) / 10.0)

    compare_snaps = sorted(all_data.keys(), reverse=True)
    print(f"\n{'Snap':>4s} {'z':>5s} | {'N_FFB':>6s} {'above':>6s} {'below':>6s} "
          f"{'f_below':>7s} {'logM_med':>8s} {'logM_min':>8s} | "
          f"{'N_FFB':>6s} {'logM_med':>8s} {'logM_min':>8s} | "
          f"{'Li_only':>7s} {'gmax_only':>9s} {'both':>6s}")
    print(f"{'':>4s} {'':>5s} | {'--- Li+ (sigmoid) ---':^40s} | "
          f"{'--- g_max ---':^28s} | {'--- overlap ---':^24s}")
    print("-" * 140)

    for snap in compare_snaps:
        z = redshifts[snap]
        logM_thr = li24_logM_thresh(z)

        if snap not in li_stats:
            continue
        ls = li_stats[snap]
        li_ffb_mask = ls['ffb']
        n_li_ffb = li_ffb_mask.sum()
        if n_li_ffb == 0:
            continue

        li_ffb_logM = ls['logM'][li_ffb_mask]
        above_line = li_ffb_logM >= logM_thr
        below_line = li_ffb_logM < logM_thr
        n_above = above_line.sum()
        n_below = below_line.sum()
        f_below = n_below / n_li_ffb if n_li_ffb > 0 else 0.0

        # g_max model (BK25)
        d = all_data[snap]
        gm_ffb = d['ffb'] == 1
        n_gm_ffb = gm_ffb.sum()

        li_logM_med = np.median(li_ffb_logM) if n_li_ffb > 0 else np.nan
        li_logM_min = li_ffb_logM.min() if n_li_ffb > 0 else np.nan
        gm_ffb_logM = d['logM'][gm_ffb]
        gm_logM_med = np.median(gm_ffb_logM) if n_gm_ffb > 0 else np.nan
        gm_logM_min = gm_ffb_logM.min() if n_gm_ffb > 0 else np.nan

        li_only = n_below
        both_approx = n_above
        gmax_only = max(0, n_gm_ffb - n_above)

        print(f"{snap:>4d} {z:>5.2f} | {n_li_ffb:>6d} {n_above:>6d} {n_below:>6d} "
              f"{f_below:>7.3f} {li_logM_med:>8.2f} {li_logM_min:>8.2f} | "
              f"{n_gm_ffb:>6d} {gm_logM_med:>8.2f} {gm_logM_min:>8.2f} | "
              f"{li_only:>7d} {gmax_only:>9d} {both_approx:>6d}")

    print()
    print("  'above/below' = Li+ FFB galaxies above/below the Li+24 threshold line")
    print("  'f_below'     = fraction of Li+ FFB galaxies selected by sigmoid spread")
    print("  'Li_only'     = sigmoid-spread galaxies (below line, no g_max analogue)")
    print("  'gmax_only'   = g_max FFB count exceeding Li+ above-line count")
    print("  'both'        = Li+ above-line FFB (overlap region)")

    # Per-snapshot detailed breakdown for snapshots with significant FFB
    print()
    print("Detailed mass breakdown (snapshots with >10 Li+ FFB galaxies):")
    print("-" * 80)
    for snap in compare_snaps:
        z = redshifts[snap]
        logM_thr = li24_logM_thresh(z)

        if snap not in li_stats:
            continue
        ls = li_stats[snap]
        li_ffb_mask = ls['ffb']
        n_li_ffb = li_ffb_mask.sum()
        if n_li_ffb < 10:
            continue

        li_ffb_logM = ls['logM'][li_ffb_mask]
        li_ffb_sm = ls['smass'][li_ffb_mask]
        above = li_ffb_logM >= logM_thr
        below = ~above

        d = all_data[snap]
        gm_ffb_logM = d['logM'][d['ffb'] == 1]
        n_gm = len(gm_ffb_logM)

        print(f"\n  Snap {snap}  z = {z:.2f}  (Li+ threshold logM = {logM_thr:.2f})")
        print(f"    Li+ FFB total:       {n_li_ffb:>6d}")
        print(f"      Above threshold:   {above.sum():>6d}  "
              f"logM range [{li_ffb_logM[above].min():.2f}, {li_ffb_logM[above].max():.2f}]"
              if above.any() else f"      Above threshold:   {0:>6d}")
        if below.any():
            print(f"      Below threshold:   {below.sum():>6d}  "
                  f"logM range [{li_ffb_logM[below].min():.2f}, {li_ffb_logM[below].max():.2f}]")
            sm_above = li_ffb_sm[above].sum()
            sm_below = li_ffb_sm[below].sum()
            sm_tot = sm_above + sm_below
            if sm_tot > 0:
                print(f"      Stellar mass above: {sm_above/sm_tot*100:>5.1f}%  "
                      f"below: {sm_below/sm_tot*100:>5.1f}%")
        else:
            print(f"      Below threshold:   {0:>6d}")
        print(f"    g_max FFB total:     {n_gm:>6d}", end="")
        if n_gm > 0:
            print(f"  logM range [{gm_ffb_logM.min():.2f}, {gm_ffb_logM.max():.2f}]")
        else:
            print()

    print()

    # ======================================================================
    # Figure 7: Side-by-side Mvir vs z — Li+ selection vs g_max selection
    # ======================================================================
    fig, (ax_li, ax_gmax) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Left panel: Li+ selected galaxies ---
    li_norm = li_ffb_all == 0
    li_ffb_mask = li_ffb_all == 1
    ax_li.scatter(li_z_all[li_norm], li_logM_all[li_norm], c='0.8', s=2,
                  alpha=0.2, rasterized=True)
    ax_li.scatter(li_z_all[li_ffb_mask], li_logM_all[li_ffb_mask], c='C3',
                  s=10, alpha=0.6, rasterized=True, zorder=5, label='FFB (Li+)')

    # Li+24 threshold line (sigmoid midpoint) + activation bands
    z_th7 = np.linspace(0, 14, 200)
    logM_li24_7 = 10.8 + np.log10(h) - 6.2 * np.log10((1 + z_th7) / 10.0)

    # Sigmoid: P(FFB) = 1/(1+exp(-(logM - logM_thresh)/sigma))
    # P=0.1 → logM = logM_thresh - ln(9)*sigma
    # P=0.9 → logM = logM_thresh + ln(9)*sigma
    sigmoid_dex_widths = [0.05, 0.07, 0.12, 0.15]
    sigmoid_alt_colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(sigmoid_dex_widths) - 1))
    ln9 = np.log(9.0)  # ~2.197
    green_idx = 0
    for sigma in sigmoid_dex_widths:
        if sigma == 0.15:
            scol = 'b'  # same as Li+24 line
            lw = 2.0
        else:
            scol = sigmoid_alt_colors[green_idx]
            green_idx += 1
            lw = 1.2
        lo = logM_li24_7 - ln9 * sigma   # P = 0.1
        hi = logM_li24_7 + ln9 * sigma   # P = 0.9
        ax_li.plot(z_th7, lo, '-', color=scol, lw=lw, alpha=0.8)
        ax_li.plot(z_th7, hi, '-', color=scol, lw=lw, alpha=0.8,
                   label=rf'$\sigma = {sigma}$ dex (10–90%)')

    ax_li.plot(z_th7, logM_li24_7, 'b--', lw=2,
               label=r'Li+24: $M \propto (1+z)^{-6.2}$')

    ax_li.set_xlabel('Redshift $z$', fontsize=13)
    ax_li.set_ylabel(r'$\log_{10}(M_{\rm vir}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax_li.set_title(r'Li+ Selected (Sigmoid)', fontsize=13)
    ax_li.set_ylim(10, 15)
    ax_li.set_xlim(0, 14)
    ax_li.legend(fontsize=8, loc='upper right')
    ax_li.grid(False)

    # --- Right panel: g_max selected galaxies ---
    gm_norm = fig6_ffb == 0
    gm_ffb = fig6_ffb == 1
    ax_gmax.scatter(fig6_z[gm_norm], fig6_logM[gm_norm], c='0.8', s=2,
                    alpha=0.2, rasterized=True)
    ax_gmax.scatter(fig6_z[gm_ffb], fig6_logM[gm_ffb], c='C3', s=10,
                    alpha=0.6, rasterized=True, zorder=5,
                    label=r'FFB ($g_{\rm max} > g_{\rm crit}$)')

    # Fixed-concentration g_max = g_crit lines
    c_fixed_7 = [3.0, 3.25, 3.5, 4.0]
    colors_c7 = plt.cm.plasma(np.linspace(0.15, 0.85, len(c_fixed_7)))
    for c_fix, col in zip(c_fixed_7, colors_c7):
        mu_c = np.log(1 + c_fix) - c_fix / (1 + c_fix)
        logM_line = []
        for zv in z_th7:
            zp1 = 1.0 + zv
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac200 = 200.0 * 4.0 * np.pi / 3.0 * rhocrit
            coeff = G_code * fac200**(2.0/3.0) * c_fix**2 / (2.0 * mu_c)
            M_code = (g_crit_code / coeff)**3
            logM_line.append(np.log10(M_code * 1e10))
        ax_gmax.plot(z_th7, logM_line, '-', color=col, lw=1.5, alpha=0.9,
                     label=f'$c = {c_fix:g}$')

    ax_gmax.set_xlabel('Redshift $z$', fontsize=13)
    ax_gmax.set_title(r'$g_{\rm max}$ Selected', fontsize=13)
    ax_gmax.set_xlim(0, 14)
    ax_gmax.legend(fontsize=8, loc='upper right')
    ax_gmax.grid(False)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'mvir_vs_z_li_vs_gmax.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 8: 2x2 comparison — Li+ models (top row), g_max models (bottom row)
    # ======================================================================
    fig, ((ax_li2, ax_nosig), (ax_bk25smooth, ax_gmax2)) = plt.subplots(
        2, 2, figsize=(16, 12), sharex=True, sharey=True)

    # --- Left panel: Li+ selected (no cut, original sigmoid) ---
    li_norm2 = li_ffb_all == 0
    li_ffb_mask2 = li_ffb_all == 1
    ax_li2.scatter(li_z_all[li_norm2], li_logM_all[li_norm2], c='0.8', s=2,
                   alpha=0.2, rasterized=True)
    ax_li2.scatter(li_z_all[li_ffb_mask2], li_logM_all[li_ffb_mask2], c='C3',
                   s=10, alpha=0.6, rasterized=True, zorder=5, label='FFB (Li+)')

    z_th8 = np.linspace(0, 14, 200)
    logM_li24_8 = 10.8 + np.log10(h) - 6.2 * np.log10((1 + z_th8) / 10.0)
    ln9 = np.log(9.0)
    lo_15 = logM_li24_8 - ln9 * 0.15
    hi_15 = logM_li24_8 + ln9 * 0.15
    ax_li2.plot(z_th8, lo_15, '-', color='b', lw=2, alpha=0.8)
    ax_li2.plot(z_th8, hi_15, '-', color='b', lw=2, alpha=0.8,
                label=r'$\sigma = 0.15$ dex (10–90%)')
    ax_li2.plot(z_th8, logM_li24_8, 'b--', lw=2,
                label=r'Li+24: $M \propto (1+z)^{-6.2}$')

    ax_li2.set_ylabel(r'$\log_{10}(M_{\rm vir}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax_li2.set_title(r'Li+ Sigmoid', fontsize=13)
    ax_li2.set_ylim(10, 15)
    ax_li2.set_xlim(0, 14)
    ax_li2.legend(fontsize=8, loc='upper right')
    ax_li2.grid(False)

    # --- Top-right panel: Li+24 no sigmoid (hard mass cutoff) ---
    nosig_norm = nosig_ffb_all == 0
    nosig_ffb_mask = nosig_ffb_all == 1
    ax_nosig.scatter(nosig_z_all[nosig_norm], nosig_logM_all[nosig_norm], c='0.8', s=2,
                     alpha=0.2, rasterized=True)
    ax_nosig.scatter(nosig_z_all[nosig_ffb_mask], nosig_logM_all[nosig_ffb_mask], c='C3',
                     s=10, alpha=0.6, rasterized=True, zorder=5,
                     label='FFB (Li+ no sigmoid)')

    ax_nosig.plot(z_th8, logM_li24_8, 'b--', lw=2,
                  label=r'Li+24: $M \propto (1+z)^{-6.2}$')

    ax_nosig.set_title(r'Li+ Hard Cutoff (no sigmoid)', fontsize=13)
    ax_nosig.set_xlim(0, 14)
    ax_nosig.legend(fontsize=8, loc='upper right')
    ax_nosig.grid(False)

    # --- Bottom-right panel: g_max selected galaxies (Ishiyama+21) ---
    gm_norm2 = fig6_ffb == 0
    gm_ffb2 = fig6_ffb == 1
    ax_gmax2.scatter(fig6_z[gm_norm2], fig6_logM[gm_norm2], c='0.8', s=2,
                     alpha=0.2, rasterized=True)
    ax_gmax2.scatter(fig6_z[gm_ffb2], fig6_logM[gm_ffb2], c='C3', s=10,
                     alpha=0.6, rasterized=True, zorder=5,
                     label=r'FFB ($g_{\rm max} > g_{\rm crit}$)')

    c_fixed_8 = [3.0, 3.25, 3.5, 4.0]
    colors_c8 = plt.cm.plasma(np.linspace(0.15, 0.85, len(c_fixed_8)))
    for c_fix, col in zip(c_fixed_8, colors_c8):
        mu_c = np.log(1 + c_fix) - c_fix / (1 + c_fix)
        logM_line = []
        for zv in z_th8:
            zp1 = 1.0 + zv
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac200 = 200.0 * 4.0 * np.pi / 3.0 * rhocrit
            coeff = G_code * fac200**(2.0/3.0) * c_fix**2 / (2.0 * mu_c)
            M_code = (g_crit_code / coeff)**3
            logM_line.append(np.log10(M_code * 1e10))
        ax_gmax2.plot(z_th8, logM_line, '-', color=col, lw=1.5, alpha=0.9,
                      label=f'$c = {c_fix:g}$')

    ax_gmax2.set_xlabel('Redshift $z$', fontsize=13)
    ax_gmax2.set_title(r'$g_{\rm max}$ Selected (Ishiyama+21)', fontsize=13)
    ax_gmax2.set_xlim(0, 14)
    ax_gmax2.legend(fontsize=8, loc='upper right')
    ax_gmax2.grid(False)

    # --- Bottom-left panel: BK25 g_max with log-normal c scatter ---
    bs_norm = bk25smooth_ffb_all == 0
    bs_ffb = bk25smooth_ffb_all == 1
    ax_bk25smooth.scatter(bk25smooth_z_all[bs_norm], bk25smooth_logM_all[bs_norm], c='0.8', s=2,
                          alpha=0.2, rasterized=True)
    ax_bk25smooth.scatter(bk25smooth_z_all[bs_ffb], bk25smooth_logM_all[bs_ffb], c='C3', s=10,
                          alpha=0.6, rasterized=True, zorder=5,
                          label=r'FFB ($g_{\rm max} > g_{\rm crit}$, $\sigma_c = 0.2$)')

    for c_fix, col in zip(c_fixed_8, colors_c8):
        mu_c = np.log(1 + c_fix) - c_fix / (1 + c_fix)
        logM_line = []
        for zv in z_th8:
            zp1 = 1.0 + zv
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac200 = 200.0 * 4.0 * np.pi / 3.0 * rhocrit
            coeff = G_code * fac200**(2.0/3.0) * c_fix**2 / (2.0 * mu_c)
            M_code = (g_crit_code / coeff)**3
            logM_line.append(np.log10(M_code * 1e10))
        ax_bk25smooth.plot(z_th8, logM_line, '-', color=col, lw=1.5, alpha=0.9,
                           label=f'$c = {c_fix:g}$')

    ax_bk25smooth.set_xlabel('Redshift $z$', fontsize=13)
    ax_bk25smooth.set_ylabel(r'$\log_{10}(M_{\rm vir}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax_bk25smooth.set_title(r'BK25 + Log-Normal $c$ Scatter ($\sigma_c = 0.2$)', fontsize=13)
    ax_bk25smooth.set_xlim(0, 14)
    ax_bk25smooth.legend(fontsize=8, loc='upper right')
    ax_bk25smooth.grid(False)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'mvir_vs_z_li_cut_gmax.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    # ======================================================================
    # Figure 9: SMF, SMD and CSFRD comparison — Li+24 vs BK25
    # ======================================================================
    print()
    print("=" * 70)
    print("Computing SMF, SMD, and CSFRD diagnostics...")
    print("=" * 70)

    # Volume in Mpc^3
    vol_frac = header['volume_fraction']
    volume = (header['box_size'] / h)**3 * vol_frac
    print(f"  Box size: {header['box_size']:.1f} Mpc/h  ->  {header['box_size']/h:.1f} Mpc")
    print(f"  Volume fraction processed: {vol_frac:.4f}")
    print(f"  Comoving volume: {volume:.2e} Mpc^3")

    # Models to compare
    diag_models = [
        {'path': LI24_MODEL, 'label': 'Li+24', 'color': 'C0', 'ls': '-'},
        {'path': MODEL_DIR, 'label': 'BK25', 'color': 'C3', 'ls': '-'},
    ]
    if os.path.exists(FFB_CUT_MODEL_DIR):
        diag_models.append(
            {'path': FFB_CUT_MODEL_DIR, 'label': 'Li+24 (with cutoff)', 'color': 'C2', 'ls': '--'})
    if os.path.exists(BK25_CONC_MODEL_DIR):
        diag_models.append(
            {'path': BK25_CONC_MODEL_DIR, 'label': r'BK25 ($V_{\rm max}/V_{\rm vir}$)', 'color': 'C5', 'ls': '--'})
    if os.path.exists(NOSIGMOID_MODEL_DIR):
        diag_models.append(
            {'path': NOSIGMOID_MODEL_DIR, 'label': 'Li+24 (no sigmoid)', 'color': 'C4', 'ls': '-.'})
    if os.path.exists(BK25_SMOOTH_MODEL_DIR):
        diag_models.append(
            {'path': BK25_SMOOTH_MODEL_DIR, 'label': r'BK25 ($\sigma_c = 0.2$)', 'color': 'C6', 'ls': '-.'})

    print(f"  Models loaded: {len(diag_models)}")
    for m in diag_models:
        print(f"    - {m['label']:30s}  ({m['path']})")

    diag_props = ['StellarMass', 'SfrDisk', 'SfrBulge']

    # --- SMF at selected redshifts ---
    smf_snaps = [20, 15, 12, 10]  # z ~ 5.3, 8.2, 10.1, 11.9
    smf_snaps = [s for s in smf_snaps if s in output_snaps]
    binwidth = 0.2
    print(f"\n  SMF snapshots: {smf_snaps}  (z = {[f'{redshifts[s]:.1f}' for s in smf_snaps]})")

    # Use common bin edges for all panels so residuals are aligned
    smf_lo, smf_hi = 5.0, 12.0
    smf_nbins = int(round((smf_hi - smf_lo) / binwidth))
    smf_centers = np.linspace(smf_lo + 0.5 * binwidth, smf_hi - 0.5 * binwidth, smf_nbins)

    # Reference model for residuals is the first model (Li+24)
    ref_label = diag_models[0]['label']
    print(f"  Residuals computed relative to: {ref_label}")

    nrows, ncols = 2, 2
    fig_smf = plt.figure(figsize=(13, 11))
    # GridSpec: each panel gets a main SMF axis + a smaller residual axis below
    gs_outer = fig_smf.add_gridspec(nrows, ncols, hspace=0.30, wspace=0.25)

    axes_main = []
    axes_res = []
    for idx in range(nrows * ncols):
        gs_inner = gs_outer[idx].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_top = fig_smf.add_subplot(gs_inner[0])
        ax_bot = fig_smf.add_subplot(gs_inner[1], sharex=ax_top)
        axes_main.append(ax_top)
        axes_res.append(ax_bot)

    for idx, snap in enumerate(smf_snaps):
        ax = axes_main[idx]
        ax_r = axes_res[idx]
        z = redshifts[snap]
        print(f"\n  --- SMF snap {snap} (z = {z:.2f}) ---")

        # Compute phi for each model on the common grid
        phi_dict = {}
        for model in diag_models:
            data = read_snap(model['path'], snap, diag_props)
            if 'StellarMass' not in data or len(data['StellarMass']) == 0:
                print(f"    {model['label']:30s}  no data")
                continue
            m_stars = data['StellarMass']
            w = m_stars > 0
            ngal = w.sum()
            if ngal == 0:
                print(f"    {model['label']:30s}  0 galaxies with M* > 0")
                continue
            log_m = np.log10(m_stars[w] * 1e10 / h)
            print(f"    {model['label']:30s}  N_gal={ngal:>8d}  "
                  f"log M* = [{log_m.min():.2f}, {log_m.max():.2f}]")
            counts, _ = np.histogram(log_m, range=(smf_lo, smf_hi), bins=smf_nbins)
            with np.errstate(divide='ignore'):
                phi = np.log10(counts / volume / binwidth)
            phi[~np.isfinite(phi)] = np.nan
            phi_dict[model['label']] = phi

            valid = np.isfinite(phi)
            ax.plot(smf_centers[valid], phi[valid], lw=2.5, color=model['color'],
                    ls=model['ls'], label=model['label'] if idx == 0 else None)

        # Residuals relative to reference
        ref_phi = phi_dict.get(ref_label)
        if ref_phi is not None:
            print(f"    Residuals (dex) vs {ref_label}:")
            for model in diag_models:
                if model['label'] == ref_label or model['label'] not in phi_dict:
                    continue
                resid = phi_dict[model['label']] - ref_phi
                both_valid = np.isfinite(resid)
                if both_valid.any():
                    med_res = np.nanmedian(resid[both_valid])
                    max_res = np.nanmax(np.abs(resid[both_valid]))
                    print(f"      {model['label']:30s}  median={med_res:+.3f}  max|resid|={max_res:.3f}")
                    ax_r.plot(smf_centers[both_valid], resid[both_valid], lw=1.8,
                              color=model['color'], ls=model['ls'])

        ax.set_title(f'$z = {z:.1f}$', fontsize=13)
        ax.set_xlim(smf_lo, smf_hi)
        ax.set_ylim(-5.5, 0)
        ax.grid(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        if idx % ncols == 0:
            ax.set_ylabel(r'$\log_{10}(\phi\ /\ {\rm Mpc}^{-3}\,{\rm dex}^{-1})$', fontsize=12)
        if idx == 0:
            ax.legend(fontsize=9)

        ax_r.axhline(0, color='grey', lw=0.8, ls='--')
        ax_r.set_ylim(-1.0, 1.0)
        ax_r.set_xlim(smf_lo, smf_hi)
        ax_r.grid(False)
        if idx >= ncols:
            ax_r.set_xlabel(r'$\log_{10}(M_\star\ /\ M_\odot)$', fontsize=12)
        if idx % ncols == 0:
            ax_r.set_ylabel(r'$\Delta\log\phi$', fontsize=11)

    # Hide any unused panels
    for idx in range(len(smf_snaps), nrows * ncols):
        axes_main[idx].set_visible(False)
        axes_res[idx].set_visible(False)

    fig_smf.suptitle('Stellar Mass Function Comparison', fontsize=15)
    outpath = os.path.join(OUTPUT_DIR, 'smf_comparison.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {outpath}")

    # --- SMD and CSFRD vs redshift ---
    print(f"\n  Computing SMD and CSFRD across {len(output_snaps)} snapshots...")

    # Build per-model arrays first so we can compute residuals
    hist_results = {}
    for model in diag_models:
        z_arr_hist = []
        smd_arr = []
        sfrd_arr = []

        for snap in output_snaps:
            z = redshifts[snap]
            data = read_snap(model['path'], snap, diag_props)
            if 'StellarMass' not in data or len(data['StellarMass']) == 0:
                continue

            m_stars = data['StellarMass'] * 1e10 / h  # Msun
            smd = np.sum(m_stars) / volume  # Msun / Mpc^3

            sfr_total = np.zeros_like(data['StellarMass'])
            if 'SfrDisk' in data:
                sfr_total += data['SfrDisk']
            if 'SfrBulge' in data:
                sfr_total += data['SfrBulge']
            sfrd = np.sum(sfr_total) / volume  # Msun/yr / Mpc^3

            z_arr_hist.append(z)
            smd_arr.append(smd)
            sfrd_arr.append(sfrd)

        z_arr_hist = np.array(z_arr_hist)
        smd_arr = np.array(smd_arr)
        sfrd_arr = np.array(sfrd_arr)

        nonzero_smd = smd_arr > 0
        nonzero_sfrd = sfrd_arr > 0

        n_smd = nonzero_smd.sum()
        n_sfrd = nonzero_sfrd.sum()
        print(f"    {model['label']:30s}  snapshots with SMD>0: {n_smd:>3d}  SFRD>0: {n_sfrd:>3d}")
        if n_smd > 0:
            print(f"      SMD  range: [{np.log10(smd_arr[nonzero_smd].min()):.2f}, "
                  f"{np.log10(smd_arr[nonzero_smd].max()):.2f}] log Msun/Mpc^3  "
                  f"(z = {z_arr_hist[nonzero_smd].max():.1f} -> {z_arr_hist[nonzero_smd].min():.1f})")
        if n_sfrd > 0:
            print(f"      SFRD range: [{np.log10(sfrd_arr[nonzero_sfrd].min()):.2f}, "
                  f"{np.log10(sfrd_arr[nonzero_sfrd].max()):.2f}] log Msun/yr/Mpc^3")

        hist_results[model['label']] = {
            'z': z_arr_hist, 'smd': smd_arr, 'sfrd': sfrd_arr,
            'color': model['color'], 'ls': model['ls'],
        }

    # Figure with residual sub-panels: 2 columns (SMD, CSFRD), each with main + residual
    fig_hist = plt.figure(figsize=(14, 7))
    gs_hist = fig_hist.add_gridspec(1, 2, wspace=0.30)

    gs_smd = gs_hist[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax_smd = fig_hist.add_subplot(gs_smd[0])
    ax_smd_r = fig_hist.add_subplot(gs_smd[1], sharex=ax_smd)

    gs_sfrd = gs_hist[1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax_sfrd = fig_hist.add_subplot(gs_sfrd[0])
    ax_sfrd_r = fig_hist.add_subplot(gs_sfrd[1], sharex=ax_sfrd)

    xlim_hist = (-0.5, max(z_arr) + 0.5)

    for model in diag_models:
        r = hist_results[model['label']]
        nz_smd = r['smd'] > 0
        nz_sfrd = r['sfrd'] > 0
        if nz_smd.any():
            ax_smd.plot(r['z'][nz_smd], np.log10(r['smd'][nz_smd]),
                        lw=2.5, color=r['color'], ls=r['ls'], label=model['label'])
        if nz_sfrd.any():
            ax_sfrd.plot(r['z'][nz_sfrd], np.log10(r['sfrd'][nz_sfrd]),
                         lw=2.5, color=r['color'], ls=r['ls'], label=model['label'])

    # Residuals relative to reference (Li+24)
    ref_r = hist_results.get(ref_label)
    if ref_r is not None:
        print(f"\n    Residuals (dex) vs {ref_label}:")
        for model in diag_models:
            if model['label'] == ref_label or model['label'] not in hist_results:
                continue
            r = hist_results[model['label']]

            # Interpolate onto the reference redshift grid for matched comparison
            # Use only redshifts present in both
            z_common = np.intersect1d(np.round(ref_r['z'], 4), np.round(r['z'], 4))

            ref_mask = np.isin(np.round(ref_r['z'], 4), z_common)
            mod_mask = np.isin(np.round(r['z'], 4), z_common)

            # SMD residual
            ref_smd_log = np.full_like(ref_r['smd'][ref_mask], np.nan)
            mod_smd_log = np.full_like(r['smd'][mod_mask], np.nan)
            nz = (ref_r['smd'][ref_mask] > 0) & (r['smd'][mod_mask] > 0)
            if nz.any():
                ref_smd_log[nz] = np.log10(ref_r['smd'][ref_mask][nz])
                mod_smd_log[nz] = np.log10(r['smd'][mod_mask][nz])
                smd_resid = mod_smd_log - ref_smd_log
                valid = np.isfinite(smd_resid)
                if valid.any():
                    z_plot = ref_r['z'][ref_mask][valid]
                    ax_smd_r.plot(z_plot, smd_resid[valid], lw=1.8, color=r['color'], ls=r['ls'])
                    med = np.median(smd_resid[valid])
                    mx = np.max(np.abs(smd_resid[valid]))
                    print(f"      {model['label']:30s}  SMD  median={med:+.3f}  max|resid|={mx:.3f}")

            # SFRD residual
            ref_sfrd_log = np.full_like(ref_r['sfrd'][ref_mask], np.nan)
            mod_sfrd_log = np.full_like(r['sfrd'][mod_mask], np.nan)
            nz2 = (ref_r['sfrd'][ref_mask] > 0) & (r['sfrd'][mod_mask] > 0)
            if nz2.any():
                ref_sfrd_log[nz2] = np.log10(ref_r['sfrd'][ref_mask][nz2])
                mod_sfrd_log[nz2] = np.log10(r['sfrd'][mod_mask][nz2])
                sfrd_resid = mod_sfrd_log - ref_sfrd_log
                valid2 = np.isfinite(sfrd_resid)
                if valid2.any():
                    z_plot2 = ref_r['z'][ref_mask][valid2]
                    ax_sfrd_r.plot(z_plot2, sfrd_resid[valid2], lw=1.8, color=r['color'], ls=r['ls'])
                    med2 = np.median(sfrd_resid[valid2])
                    mx2 = np.max(np.abs(sfrd_resid[valid2]))
                    print(f"      {model['label']:30s}  SFRD median={med2:+.3f}  max|resid|={mx2:.3f}")

    # Format SMD panels
    plt.setp(ax_smd.get_xticklabels(), visible=False)
    ax_smd.set_ylabel(r'$\log_{10}(\rho_\star\ /\ M_\odot\,{\rm Mpc}^{-3})$', fontsize=13)
    ax_smd.set_title('Stellar Mass Density', fontsize=13)
    ax_smd.legend(fontsize=9)
    ax_smd.grid(False)
    ax_smd.set_xlim(*xlim_hist)
    ax_smd.set_ylim(3.5, 9.0)
    ax_smd_r.axhline(0, color='grey', lw=0.8, ls='--')
    ax_smd_r.set_ylim(-1.0, 1.0)
    ax_smd_r.set_xlabel('Redshift $z$', fontsize=13)
    ax_smd_r.set_ylabel(r'$\Delta\log\rho_\star$', fontsize=11)
    ax_smd_r.grid(False)

    # Format SFRD panels
    plt.setp(ax_sfrd.get_xticklabels(), visible=False)
    ax_sfrd.set_ylabel(r'$\log_{10}(\dot{\rho}_\star\ /\ M_\odot\,{\rm yr}^{-1}\,{\rm Mpc}^{-3})$', fontsize=13)
    ax_sfrd.set_title('Cosmic SFR Density', fontsize=13)
    ax_sfrd.legend(fontsize=9)
    ax_sfrd.grid(False)
    ax_sfrd.set_xlim(*xlim_hist)
    ax_sfrd.set_ylim(-4.75,-0.8)
    ax_sfrd_r.axhline(0, color='grey', lw=0.8, ls='--')
    ax_sfrd_r.set_ylim(-1.0, 1.0)
    ax_sfrd_r.set_xlabel('Redshift $z$', fontsize=13)
    ax_sfrd_r.set_ylabel(r'$\Delta\log\dot{\rho}_\star$', fontsize=11)
    ax_sfrd_r.grid(False)

    outpath = os.path.join(OUTPUT_DIR, 'smd_sfrd_comparison.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {outpath}")

    # ======================================================================
    # Figure 10: BK25 FFB probability vs halo mass — does c-scatter ≈ sigmoid?
    # ======================================================================
    print()
    print("=" * 70)
    print("Figure 10: BK25 log-normal c-scatter FFB probability vs Li+24 sigmoid")
    print("=" * 70)

    try:
        from colossus.cosmology import cosmology as colossus_cosmo
        from colossus.halo import concentration as colossus_conc
        colossus_cosmo.setCosmology('mill_fig10', flat=True, H0=h*100,
                                     Om0=Om, Ob0=0.045, sigma8=0.9, ns=1.0,
                                     relspecies=False)

        fig10_redshifts = [6, 8, 10, 12]
        sigma_c = 0.2  # ln(c) scatter
        N_MC = 10000   # Monte Carlo draws per mass bin
        logM_grid = np.linspace(9.0, 13.5, 200)  # log10(M / [Msun/h])
        rng = np.random.default_rng(42)

        fig, axes = plt.subplots(1, len(fig10_redshifts), figsize=(6*len(fig10_redshifts), 5),
                                 sharey=True)

        for ax, z_val in zip(axes, fig10_redshifts):
            print(f"  z = {z_val} ...")

            # --- BK25 FFB probability via Monte Carlo c-scatter ---
            M_arr = 10**logM_grid  # Msun/h
            c_mean = colossus_conc.concentration(M_arr, '200c', z_val, model='ishiyama21')
            c_mean = np.clip(c_mean, 1.0, None)

            # Virial radius from M200c definition: M = (4π/3) * 200 * ρ_crit * R^3
            zp1 = 1.0 + z_val
            H_sq = Hubble_code**2 * (Om * zp1**3 + (1 - Om))
            rhocrit = 3.0 * H_sq / (8 * np.pi * G_code)
            fac = 1.0 / (200.0 * 4 * np.pi / 3.0 * rhocrit)
            Mvir_code = M_arr / 1e10  # code units (10^10 Msun/h)
            Rvir_code = (Mvir_code * fac)**(1./3.)

            # g_crit in code units (already computed above as g_crit_code)
            # Monte Carlo: draw N_MC concentrations per mass, compute FFB fraction
            p_ffb_bk25 = np.zeros(len(logM_grid))
            for j in range(len(logM_grid)):
                # log-normal scatter: ln(c) = ln(c_mean) + sigma_c * N(0,1)
                ln_c_draws = np.log(c_mean[j]) + sigma_c * rng.standard_normal(N_MC)
                c_draws = np.exp(ln_c_draws)
                c_draws = np.clip(c_draws, 1.0, None)

                g_vir = G_code * Mvir_code[j] / Rvir_code[j]**2
                mu_c = np.log(1.0 + c_draws) - c_draws / (1.0 + c_draws)
                g_max_draws = g_vir * c_draws**2 / (2.0 * mu_c)
                p_ffb_bk25[j] = np.mean(g_max_draws > g_crit_code)

            # --- Li+24 sigmoid ---
            # M_thresh in code units: log10(M_code) = 0.8 + log10(h) - 6.2*log10((1+z)/10)
            z_norm = (1.0 + z_val) / 10.0
            log_Mthresh_code = 0.8 + np.log10(h) - 6.2 * np.log10(z_norm)
            Mthresh_code = 10**log_Mthresh_code
            # Convert to Msun/h for comparison on same x-axis
            log_Mthresh_Msun_h = log_Mthresh_code + 10.0
            # Sigmoid: f_ffb = 1 / (1 + exp(-log10(M/M_thresh) / 0.15))
            # Working in log10(Msun/h):
            x_sigmoid = (logM_grid - log_Mthresh_Msun_h) / 0.15
            f_sigmoid = 1.0 / (1.0 + np.exp(-x_sigmoid))

            # --- BK25 sharp threshold (σ_c = 0) ---
            p_ffb_sharp = np.zeros(len(logM_grid))
            for j in range(len(logM_grid)):
                g_vir = G_code * Mvir_code[j] / Rvir_code[j]**2
                mu_c = np.log(1.0 + c_mean[j]) - c_mean[j] / (1.0 + c_mean[j])
                g_max_val = g_vir * c_mean[j]**2 / (2.0 * mu_c)
                p_ffb_sharp[j] = 1.0 if g_max_val > g_crit_code else 0.0

            # --- Plot ---
            ax.plot(logM_grid, p_ffb_bk25, '-', color='dodgerblue', lw=3,
                    label=r'BK25 ($\sigma_c=0.2$)')
            ax.plot(logM_grid, f_sigmoid, '--', color='black', lw=2.5,
                    label=r'Li+24 sigmoid ($\Delta=0.15$ dex)')
            ax.plot(logM_grid, p_ffb_sharp, ':', color='darkgreen', lw=2,
                    label='BK25 sharp')
            ax.axvline(log_Mthresh_Msun_h, color='firebrick', ls='--', lw=1.5, alpha=0.6,
                       label=r'$M_{\rm thresh}$ (Li+24)')

            ax.set_xlabel(r'$\log_{10}\ M_{\rm vir}\ [M_\odot/h]$', fontsize=13)
            ax.set_title(f'$z = {z_val}$', fontsize=14)
            ax.set_xlim(9.5, 13.0)
            ax.set_ylim(-0.05, 1.05)
            if ax == axes[0]:
                ax.set_ylabel('P(FFB)', fontsize=13)
                ax.legend(fontsize=10, loc='upper left')

        fig.suptitle(r'BK25 log-normal $c$-scatter vs Li+24 sigmoid: FFB probability',
                     fontsize=15, y=1.02)
        fig.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, 'ffb_probability_sigmoid_vs_scatter.pdf')
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved {outpath}")

    except ImportError:
        print("  [Colossus not available — skipping Figure 10]")

    print()
    print("=" * 70)
    print("Validation complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()

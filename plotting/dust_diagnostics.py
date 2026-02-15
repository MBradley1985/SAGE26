#!/usr/bin/env python
"""
Dust Diagnostics for SAGE26
============================
Comprehensive diagnostics to verify and visualise the dust model,
including the yield-table-driven production (MetalYieldsOn=1).

Produces:
  - Detailed terminal statistics (mass-binned table, sanity checks,
    MW-mass selection, galaxy type breakdown, redshift evolution,
    cosmic dust budget)
  - 4x2 panel PDF with:
      1. Cosmic dust density evolution (rho_dust vs z)
      2. Dust-to-gas ratio vs metallicity (DtG–Z relation)
      3. Dust-to-metal ratio vs stellar mass
      4. Dust mass function at z=0
      5. Dust reservoir breakdown vs stellar mass
      6. Dust-to-gas ratio vs stellar mass
      7. Specific dust mass (M_dust/M_star) vs stellar mass
      8. Dust-to-gas ratio evolution at fixed mass bins

  Observational data overlaid from:
    - Rémy-Ruyer et al. (2014) — DtG, DtM, Mdust individual galaxies
    - Dunne et al. (2011) — dust mass function, cosmic dust density
    - Vlahakis et al. (2005) — dust mass function
    - Clemens et al. (2013) — dust mass function
    - Ménard & Fukugita (2012) — cosmic dust density
    - DustPedia / Nersesian et al. (2019) — Mdust/Mstar

Usage:
  python plotting/dust_diagnostics.py [--dir output/millennium/]
  python plotting/dust_diagnostics.py --dir output/millennium/ --compare output/millennium_noffb/
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import os
import sys

# ========================== USER OPTIONS ==========================

# Simulation details
Hubble_h = 0.73
BoxSize  = 62.5         # Mpc/h
VolumeFraction = 1.0
FirstSnap = 0
LastSnap  = 63
HYDROGEN_MASS_FRAC = 0.76

# Redshift list for Millennium (snaps 0–63)
redshifts = np.array([
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
    14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
     7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
     3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
     2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
     1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
     0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
     0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000])

# Plotting style – dark theme matching allresults-local.py
plt.rcParams["figure.dpi"]  = 120
plt.rcParams["font.size"]   = 12
plt.rcParams['figure.facecolor']  = 'black'
plt.rcParams['axes.facecolor']    = 'black'
plt.rcParams['axes.edgecolor']    = 'white'
plt.rcParams['xtick.color']       = 'white'
plt.rcParams['ytick.color']       = 'white'
plt.rcParams['axes.labelcolor']   = 'white'
plt.rcParams['axes.titlecolor']   = 'white'
plt.rcParams['text.color']        = 'white'
plt.rcParams['legend.facecolor']  = 'black'
plt.rcParams['legend.edgecolor']  = 'white'

OutputFormat = '.pdf'

# ==================================================================
# Observational Data
# ==================================================================

def load_remy_ruyer_2014():
    """Load Rémy-Ruyer+2014 galaxy sample from dusty-sage analysis directory.
    Returns dict with keys: logMstar, logMdust, logSFR, Z_12logOH, logMgas"""
    datafile = os.path.join(os.path.dirname(__file__),
                            './data/remy-ruyer.master.dat')
    if not os.path.exists(datafile):
        # Try relative to workspace root
        datafile = os.path.expanduser(
            './data/remy-ruyer.master.dat')
    if not os.path.exists(datafile):
        return None

    logMstar = []
    logMdust_AC = []   # amorphous carbon
    Z_12logOH = []
    logSFR = []
    logMgas = []

    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                HI = float(parts[1])
                Z = float(parts[3])
                H2_MW = float(parts[4])
                logMd_AC = float(parts[8])
                logMs = float(parts[9])
                lSFR = float(parts[10])
            except (ValueError, IndexError):
                continue
            if logMs <= 0 or logMd_AC <= 0 or Z <= 0:
                continue
            gas_mass = HI + H2_MW
            if gas_mass > 0:
                logMgas.append(np.log10(gas_mass))
            else:
                logMgas.append(np.nan)
            logMstar.append(logMs)
            logMdust_AC.append(logMd_AC)
            Z_12logOH.append(Z)
            logSFR.append(lSFR)

    if len(logMstar) == 0:
        return None

    return {
        'logMstar': np.array(logMstar),
        'logMdust': np.array(logMdust_AC),
        'logSFR': np.array(logSFR),
        'Z_12logOH': np.array(Z_12logOH),
        'logMgas': np.array(logMgas),
    }


def obs_dmf_z0():
    """Dust mass function observations at z=0.
    Returns dict of datasets, each with logMd, logphi, errhi, errlo."""

    # Vlahakis et al. 2005 (SCUBA Local Universe Galaxy Survey)
    vlahakis = {
        'logMd': np.array([6.289, 6.536, 6.784, 7.032, 7.287, 7.528, 7.776, 8.024, 8.289, 8.536]),
        'logphi': np.array([-1.389, -1.389, -1.504, -1.601, -1.679, -1.873, -2.105, -2.414, -2.782, -3.248]),
        'errhi': np.array([0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.215, 0.293, 0.508]),
        'errlo': np.array([0.137, 0.098, 0.098, 0.098, 0.117, 0.137, 0.156, 0.235, 0.508, 0.781]),
    }

    # Clemens et al. 2013 (Planck Early Release Compact Source Catalogue)
    clemens = {
        'logMd': np.array([6.405, 6.653, 6.949, 7.238, 7.478, 7.735, 7.982, 8.271, 8.560, 9.096]),
        'logphi': np.array([-1.536, -1.679, -1.873, -1.815, -1.893, -2.185, -2.555, -3.081, -3.567, -4.757]),
        'errhi': np.array([0.098, 0.098, 0.098, 0.098, 0.098, 0.117, 0.156, 0.195, 0.273, 0.742]),
        'errlo': np.array([0.117, 0.137, 0.137, 0.117, 0.098, 0.137, 0.195, 0.332, 0.547, 1.445]),
    }

    # Dunne et al. 2011 (H-ATLAS)
    dunne = {
        'logMd': np.array([6.071, 6.321, 6.571, 6.821, 7.071, 7.321, 7.571, 7.821, 8.071, 8.160, 8.409]),
        'logphi': np.array([-1.199, -1.102, -1.258, -1.316, -1.473, -1.717, -2.059, -2.362, -2.921, -3.186, -3.733]),
        'errhi': np.array([0.098, 0.078, 0.059, 0.059, 0.059, 0.039, 0.039, 0.078, 0.098, 0.234, 0.469]),
        'errlo': np.array([0.098, 0.059, 0.059, 0.059, 0.059, 0.039, 0.059, 0.078, 0.117, 0.313, 0.781]),
    }

    return {'Vlahakis+05': vlahakis, 'Clemens+13': clemens, 'Dunne+11': dunne}


def obs_cosmic_dust_density():
    """Cosmic dust mass density observations.
    Returns dict of datasets with z, logrho, errhi, errlo."""

    # Dunne et al. 2011
    dunne = {
        'z': np.array([0.052, 0.155, 0.251, 0.36, 0.45]),
        'logrho': np.array([4.986, 5.186, 5.316, 5.488, 5.353]),
        'errhi': np.array([0.054, 0.066, 0.049, 0.128, 0.089]),
        'errlo': np.array([0.08, 0.066, 0.078, 0.148, 0.181]),
    }

    # Ménard & Fukugita 2012
    menard = {
        'z': np.array([0.661, 0.976, 1.300, 1.618, 1.937]),
        'logrho': np.array([5.749, 5.767, 5.743, 5.631, 5.613]),
        'errhi': np.array([0.114, 0.092, 0.074, 0.062, 0.116]),
        'errlo': np.array([0.114, 0.092, 0.074, 0.062, 0.116]),
    }

    return {'Dunne+11': dunne, 'Ménard+12': menard}


def obs_dustpedia_nersesian2019():
    """DustPedia/Nersesian+2019 binned data points.
    Returns dict with logMstar, logMdust arrays (16 points)."""
    return {
        'logMstar': np.array([8.14, 8.31, 8.69, 8.76, 9.04, 9.26,
                              9.38, 9.62, 9.96, 10.08, 10.22, 10.56,
                              10.57, 10.67, 10.89, 10.96]),
        'logMdust': np.array([4.74, 5.46, 5.68, 5.48, 6.39, 6.36,
                              6.62, 6.86, 6.95, 7.14, 7.22, 7.45,
                              7.32, 7.42, 7.55, 7.51]),
        'logMstar_err': np.array([0.27, 0.17, 0.15, 0.28, 0.14, 0.11,
                                  0.16, 0.13, 0.14, 0.11, 0.13, 0.15,
                                  0.13, 0.11, 0.12, 0.14]),
        'logMdust_err': np.array([0.43, 0.33, 0.36, 0.47, 0.39, 0.38,
                                  0.32, 0.34, 0.38, 0.27, 0.28, 0.30,
                                  0.42, 0.24, 0.27, 0.36]),
    }


# ==================================================================
# Helper: read parameter from one snapshot across all file chunks
# ==================================================================

def read_all_files(dirpath, snap_str, param):
    """Read *param* from every model_*.hdf5 file for a given snapshot."""
    arrays = []
    i = 0
    while True:
        fname = os.path.join(dirpath, f'model_{i}.hdf5')
        if not os.path.exists(fname):
            break
        with h5.File(fname, 'r') as f:
            if snap_str in f and param in f[snap_str]:
                arrays.append(np.array(f[snap_str][param]))
        i += 1
    if not arrays:
        return None
    return np.concatenate(arrays)


# ==================================================================
# MAIN
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description='SAGE26 dust diagnostics')
    parser.add_argument('--dir', default='./output/millennium/',
                        help='Directory containing model_*.hdf5 files')
    parser.add_argument('--compare', default=None,
                        help='Optional second output directory to compare')
    args = parser.parse_args()

    DirName = args.dir
    CompareDir = args.compare

    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    # Determine output directory (same as input data directory)
    OutputDir = DirName

    # ---- Check that files exist ----
    if not os.path.exists(os.path.join(DirName, 'model_0.hdf5')):
        print(f'ERROR: No model_0.hdf5 found in {DirName}')
        sys.exit(1)

    # ==================================================================
    # 0. Read z=0 data
    # ==================================================================
    snap_str = f'Snap_{LastSnap}'

    StellarMass   = read_all_files(DirName, snap_str, 'StellarMass')   * 1e10 / Hubble_h
    ColdGas       = read_all_files(DirName, snap_str, 'ColdGas')       * 1e10 / Hubble_h
    MetalsColdGas = read_all_files(DirName, snap_str, 'MetalsColdGas') * 1e10 / Hubble_h
    HotGas        = read_all_files(DirName, snap_str, 'HotGas')        * 1e10 / Hubble_h
    MetalsHotGas  = read_all_files(DirName, snap_str, 'MetalsHotGas')  * 1e10 / Hubble_h

    ColdDust_arr    = read_all_files(DirName, snap_str, 'ColdDust')
    HotDust_arr     = read_all_files(DirName, snap_str, 'HotDust')
    CGMDust_arr     = read_all_files(DirName, snap_str, 'CGMDust')
    EjectedDust_arr = read_all_files(DirName, snap_str, 'EjectedDust')

    if ColdDust_arr is None:
        print('ERROR: No ColdDust found in output — DustOn may be 0.')
        sys.exit(1)

    ColdDust    = ColdDust_arr    * 1e10 / Hubble_h
    HotDust     = HotDust_arr     * 1e10 / Hubble_h
    CGMDust     = CGMDust_arr     * 1e10 / Hubble_h if CGMDust_arr is not None else np.zeros(len(ColdDust))
    EjectedDust = EjectedDust_arr * 1e10 / Hubble_h
    TotalDust   = ColdDust + HotDust + CGMDust + EjectedDust

    # Also read SFR, Type, Vvir for richer statistics
    SfrDisk_arr  = read_all_files(DirName, snap_str, 'SfrDisk')
    SfrBulge_arr = read_all_files(DirName, snap_str, 'SfrBulge')
    Type_arr     = read_all_files(DirName, snap_str, 'Type')
    Vvir_arr     = read_all_files(DirName, snap_str, 'Vvir')

    if SfrDisk_arr is not None and SfrBulge_arr is not None:
        SFR = SfrDisk_arr + SfrBulge_arr
    else:
        SFR = np.zeros(len(StellarMass))

    if Type_arr is None:
        Type_arr = np.zeros(len(StellarMass), dtype=int)
    if Vvir_arr is None:
        Vvir_arr = np.zeros(len(StellarMass))

    Ngal = len(StellarMass)

    # Safe log10 of stellar mass (avoids log10(0) warnings)
    with np.errstate(divide='ignore', invalid='ignore'):
        logMstar = np.where(StellarMass > 0, np.log10(StellarMass), -99.0)

    # ==================================================================
    # TERMINAL STATISTICS BLOCK
    # ==================================================================
    print(f'\n{"="*70}')
    print(f'  SAGE26 Dust Diagnostics — {DirName}')
    print(f'{"="*70}')

    # --- 1. Basic galaxy counts ---
    print(f'\n  [1] GALAXY COUNTS (z = 0)')
    print(f'  {"—"*40}')
    print(f'  Total galaxies:                  {Ngal}')
    print(f'  Centrals (Type=0):               {(Type_arr == 0).sum()}')
    print(f'  Satellites (Type=1):             {(Type_arr == 1).sum()}')
    print(f'  Orphans (Type=2):                {(Type_arr == 2).sum()}')
    print(f'  With StellarMass > 0:            {(StellarMass > 0).sum()}')
    print(f'  With ColdGas > 0:                {(ColdGas > 0).sum()}')
    n_sf = (SFR > 0).sum()
    print(f'  Star-forming (SFR > 0):          {n_sf}  ({100*n_sf/Ngal:.1f}%)')

    # --- 2. Dust reservoir overview ---
    print(f'\n  [2] DUST RESERVOIR OVERVIEW')
    print(f'  {"—"*40}')
    print(f'  With ColdDust > 0:               {(ColdDust > 0).sum():>6}  ({100*(ColdDust>0).sum()/Ngal:.1f}%)')
    print(f'  With HotDust > 0:                {(HotDust > 0).sum():>6}  ({100*(HotDust>0).sum()/Ngal:.1f}%)')
    print(f'  With CGMDust > 0:                {(CGMDust > 0).sum():>6}  ({100*(CGMDust>0).sum()/Ngal:.1f}%)')
    print(f'  With EjectedDust > 0:            {(EjectedDust > 0).sum():>6}  ({100*(EjectedDust>0).sum()/Ngal:.1f}%)')
    print(f'  With any dust > 0:               {(TotalDust > 0).sum():>6}  ({100*(TotalDust>0).sum()/Ngal:.1f}%)')

    # --- 3. Sanity checks ---
    print(f'\n  [3] SANITY CHECKS')
    print(f'  {"—"*40}')
    n_neg_cold = (ColdDust < 0).sum()
    n_neg_hot  = (HotDust  < 0).sum()
    n_neg_cgm  = (CGMDust  < 0).sum()
    n_neg_ej   = (EjectedDust < 0).sum()
    print(f'  Negative ColdDust:               {n_neg_cold}  {"✓" if n_neg_cold == 0 else "⚠ WARNING"}')
    print(f'  Negative HotDust:                {n_neg_hot}  {"✓" if n_neg_hot == 0 else "⚠ WARNING"}')
    print(f'  Negative CGMDust:                {n_neg_cgm}  {"✓" if n_neg_cgm == 0 else "⚠ WARNING"}')
    print(f'  Negative EjectedDust:            {n_neg_ej}  {"✓" if n_neg_ej == 0 else "⚠ WARNING"}')

    if n_neg_hot > 0:
        worst_neg = HotDust[HotDust < 0].min()
        print(f'    Most negative HotDust:         {worst_neg:.4e} Msun')

    # Dust exceeding metals?
    w_check = np.where((ColdDust > 0) & (MetalsColdGas > 0))[0]
    n_dust_gt_metals = (ColdDust[w_check] > MetalsColdGas[w_check]).sum() if len(w_check) > 0 else 0
    print(f'  ColdDust > MetalsColdGas:        {n_dust_gt_metals}  {"✓" if n_dust_gt_metals == 0 else "⚠ WARNING"}')

    # Dust exceeding cold gas?
    n_dust_gt_gas = ((ColdDust > 0) & (ColdDust > ColdGas)).sum()
    print(f'  ColdDust > ColdGas:              {n_dust_gt_gas}  {"✓" if n_dust_gt_gas == 0 else "⚠ WARNING"}')

    # NaN/Inf checks
    n_nan = np.isnan(ColdDust).sum() + np.isnan(HotDust).sum() + np.isnan(CGMDust).sum() + np.isnan(EjectedDust).sum()
    n_inf = np.isinf(ColdDust).sum() + np.isinf(HotDust).sum() + np.isinf(CGMDust).sum() + np.isinf(EjectedDust).sum()
    print(f'  NaN in any dust array:           {n_nan}  {"✓" if n_nan == 0 else "⚠ WARNING"}')
    print(f'  Inf in any dust array:           {n_inf}  {"✓" if n_inf == 0 else "⚠ WARNING"}')

    # --- 4. Global ratios (star-forming galaxies) ---
    print(f'\n  [4] GLOBAL DUST RATIOS (ColdGas > 10^4 Msun, ColdDust > 0)')
    print(f'  {"—"*40}')
    w_sf = np.where((ColdGas > 1e4) & (ColdDust > 0) & (MetalsColdGas > 0))[0]
    if len(w_sf) > 0:
        dtg_vals = ColdDust[w_sf] / ColdGas[w_sf]
        dtm_vals = ColdDust[w_sf] / MetalsColdGas[w_sf]
        print(f'  N galaxies in selection:         {len(w_sf)}')
        print(f'                                   16th     median   84th     mean')
        print(f'  Dust-to-gas ratio:               {np.percentile(dtg_vals, 16):.4f}   {np.median(dtg_vals):.4f}   {np.percentile(dtg_vals, 84):.4f}   {np.mean(dtg_vals):.4f}')
        print(f'  Dust-to-metal ratio:             {np.percentile(dtm_vals, 16):.4f}   {np.median(dtm_vals):.4f}   {np.percentile(dtm_vals, 84):.4f}   {np.mean(dtm_vals):.4f}')
        print(f'  Reference: MW DtG ~ 0.01, DtM ~ 0.5')
    else:
        print(f'  No star-forming galaxies with dust found!')

    # --- 5. Mass-binned statistics table ---
    print(f'\n  [5] MASS-BINNED STATISTICS (z = 0)')
    print(f'  {"—"*40}')
    mass_edges = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
    header = f'  {"log M*":>10s}  {"N_gal":>6s}  {"N_dust":>6s}  {"med DtG":>9s}  {"med DtM":>9s}  {"med log M_d":>11s}  {"med sSFR":>9s}'
    print(header)
    print(f'  {"—"*len(header)}')

    for i in range(len(mass_edges) - 1):
        lo, hi = mass_edges[i], mass_edges[i+1]
        label = f'{lo:.1f}-{hi:.1f}'
        wm = np.where((logMstar >= lo) & (logMstar < hi))[0]
        n_gal = len(wm)
        wd = np.where((logMstar >= lo) & (logMstar < hi) &
                       (ColdDust > 0) & (ColdGas > 1e4) & (MetalsColdGas > 0))[0]
        n_dust = len(wd)

        if n_dust > 5:
            med_dtg = np.median(ColdDust[wd] / ColdGas[wd])
            med_dtm = np.median(ColdDust[wd] / MetalsColdGas[wd])
            med_logMd = np.median(np.log10(ColdDust[wd]))
        else:
            med_dtg = med_dtm = med_logMd = np.nan

        # sSFR
        ws = np.where((logMstar >= lo) & (logMstar < hi) & (SFR > 0))[0]
        med_ssfr = np.median(np.log10(SFR[ws] / StellarMass[ws])) if len(ws) > 5 else np.nan

        print(f'  {label:>10s}  {n_gal:>6d}  {n_dust:>6d}  {med_dtg:>9.4f}  {med_dtm:>9.4f}  {med_logMd:>11.2f}  {med_ssfr:>9.2f}')

    # --- 6. Central vs Satellite breakdown ---
    print(f'\n  [6] CENTRAL vs SATELLITE DUST')
    print(f'  {"—"*40}')
    for gtype, gname in [(0, 'Centrals'), (1, 'Satellites'), (2, 'Orphans')]:
        wt = np.where((Type_arr == gtype) & (ColdGas > 1e4) & (ColdDust > 0) & (MetalsColdGas > 0))[0]
        if len(wt) > 5:
            dtg_t = np.median(ColdDust[wt] / ColdGas[wt])
            dtm_t = np.median(ColdDust[wt] / MetalsColdGas[wt])
            md_t  = np.median(np.log10(ColdDust[wt]))
            print(f'  {gname:12s}  N={len(wt):>5d}  med DtG={dtg_t:.4f}  med DtM={dtm_t:.4f}  med log(Md)={md_t:.2f}')
        else:
            print(f'  {gname:12s}  N={len(wt):>5d}  (too few for statistics)')

    # --- 7. Milky Way analogues ---
    print(f'\n  [7] MILKY WAY ANALOGUES (10.5 < log M* < 11.0, SFR > 0)')
    print(f'  {"—"*40}')
    w_mw = np.where((logMstar >= 10.5) & (logMstar < 11.0) &
                     (SFR > 0) & (ColdDust > 0) & (ColdGas > 0) & (MetalsColdGas > 0))[0]
    if len(w_mw) > 3:
        print(f'  N MW-like galaxies:              {len(w_mw)}')
        mw_dtg = ColdDust[w_mw] / ColdGas[w_mw]
        mw_dtm = ColdDust[w_mw] / MetalsColdGas[w_mw]
        mw_md  = ColdDust[w_mw]
        mw_sfr = SFR[w_mw]
        print(f'  Median DtG:                      {np.median(mw_dtg):.4f}  (MW obs ~ 0.01)')
        print(f'  Median DtM:                      {np.median(mw_dtm):.4f}  (MW obs ~ 0.5)')
        print(f'  Median log(M_dust/Msun):         {np.median(np.log10(mw_md)):.2f}  (MW obs ~ 7.7)')
        print(f'  Median SFR (Msun/yr):            {np.median(mw_sfr):.2f}  (MW obs ~ 1-3)')
        print(f'  Median log(M_cold_gas/Msun):     {np.median(np.log10(ColdGas[w_mw])):.2f}  (MW obs ~ 10.0)')
        mw_Z = MetalsColdGas[w_mw] / ColdGas[w_mw]
        print(f'  Median Z_gas/Z_sun:              {np.median(mw_Z / 0.02):.2f}  (MW obs ~ 1.0)')
    else:
        print(f'  Too few MW analogues found ({len(w_mw)})')

    # --- 8. Cosmic dust budget ---
    print(f'\n  [8] COSMIC DUST BUDGET (z = 0)')
    print(f'  {"—"*40}')
    rho_dust_cold    = ColdDust.sum()    / volume
    rho_dust_hot     = np.maximum(HotDust, 0).sum() / volume
    rho_dust_cgm     = np.maximum(CGMDust, 0).sum() / volume
    rho_dust_ejected = EjectedDust.sum() / volume
    rho_dust_total   = rho_dust_cold + rho_dust_hot + rho_dust_cgm + rho_dust_ejected
    rho_crit = 2.775e11  # h^2 Msun/Mpc^3

    rho_metals_cold = MetalsColdGas.sum() / volume
    rho_metals_hot  = MetalsHotGas.sum()  / volume
    rho_stars       = StellarMass.sum()   / volume
    rho_cold_gas    = ColdGas.sum()       / volume
    rho_hot_gas     = HotGas.sum()        / volume

    print(f'  Dust densities (Msun/Mpc^3):')
    print(f'    Cold (ISM):       {rho_dust_cold:>12.3e}  ({100*rho_dust_cold/rho_dust_total:>5.1f}% of total dust)')
    print(f'    Hot  (ICM):       {rho_dust_hot:>12.3e}  ({100*rho_dust_hot/rho_dust_total:>5.1f}%)')
    print(f'    CGM:              {rho_dust_cgm:>12.3e}  ({100*rho_dust_cgm/rho_dust_total:>5.1f}%)')
    print(f'    Ejected:          {rho_dust_ejected:>12.3e}  ({100*rho_dust_ejected/rho_dust_total:>5.1f}%)')
    print(f'    Total:            {rho_dust_total:>12.3e}')
    print(f'  Omega_dust:           {rho_dust_total / rho_crit:.3e}   (obs ~ 1-5 x 10^-6)')
    print()
    print(f'  Context densities (Msun/Mpc^3):')
    print(f'    Cold gas:         {rho_cold_gas:>12.3e}')
    print(f'    Hot gas:          {rho_hot_gas:>12.3e}')
    print(f'    Metals (cold):    {rho_metals_cold:>12.3e}')
    print(f'    Metals (hot):     {rho_metals_hot:>12.3e}')
    print(f'    Stellar mass:     {rho_stars:>12.3e}')
    print()
    print(f'  Global ratios:')
    cosmic_dtg = rho_dust_cold / rho_cold_gas if rho_cold_gas > 0 else 0
    cosmic_dtm = rho_dust_cold / rho_metals_cold if rho_metals_cold > 0 else 0
    cosmic_dust_frac_metals = rho_dust_total / (rho_metals_cold + rho_metals_hot) if (rho_metals_cold + rho_metals_hot) > 0 else 0
    print(f'    Cosmic DtG (cold):              {cosmic_dtg:.4f}')
    print(f'    Cosmic DtM (cold):              {cosmic_dtm:.4f}')
    print(f'    Dust / all metals:              {cosmic_dust_frac_metals:.4f}')

    # --- 9. Dust mass distribution percentiles ---
    print(f'\n  [9] DUST MASS DISTRIBUTION (galaxies with ColdDust > 0)')
    print(f'  {"—"*40}')
    w_any = np.where(ColdDust > 0)[0]
    if len(w_any) > 0:
        logMd = np.log10(ColdDust[w_any])
        pcts = [5, 16, 25, 50, 75, 84, 95]
        vals = np.percentile(logMd, pcts)
        pct_str = '  '.join([f'{p}th={v:.2f}' for p, v in zip(pcts, vals)])
        print(f'  log(M_cold_dust / Msun):')
        print(f'    {pct_str}')

    w_any_t = np.where(TotalDust > 0)[0]
    if len(w_any_t) > 0:
        logMdT = np.log10(TotalDust[w_any_t])
        vals_t = np.percentile(logMdT, pcts)
        pct_str_t = '  '.join([f'{p}th={v:.2f}' for p, v in zip(pcts, vals_t)])
        print(f'  log(M_total_dust / Msun):')
        print(f'    {pct_str_t}')

    # --- 10. Extreme galaxies ---
    print(f'\n  [10] EXTREME / NOTABLE GALAXIES')
    print(f'  {"—"*40}')
    if (ColdDust > 0).sum() > 0:
        imax = np.argmax(ColdDust)
        print(f'  Most dusty (cold) galaxy:')
        print(f'    log M_dust = {np.log10(ColdDust[imax]):.2f},  log M* = {np.log10(max(StellarMass[imax],1)):.2f},  '
              f'log M_gas = {np.log10(max(ColdGas[imax],1)):.2f},  Type = {Type_arr[imax]}')
        if ColdGas[imax] > 0:
            print(f'    DtG = {ColdDust[imax]/ColdGas[imax]:.4f}')
        if MetalsColdGas[imax] > 0:
            print(f'    DtM = {ColdDust[imax]/MetalsColdGas[imax]:.4f}')

    # Dustiest galaxy by DtG
    w_dtg_check = np.where((ColdGas > 1e6) & (ColdDust > 0))[0]
    if len(w_dtg_check) > 0:
        dtg_all = ColdDust[w_dtg_check] / ColdGas[w_dtg_check]
        imax_dtg = w_dtg_check[np.argmax(dtg_all)]
        print(f'  Highest DtG (ColdGas > 10^6):')
        print(f'    DtG = {ColdDust[imax_dtg] / ColdGas[imax_dtg]:.4f},  '
              f'log M* = {np.log10(max(StellarMass[imax_dtg],1)):.2f},  '
              f'log M_gas = {np.log10(ColdGas[imax_dtg]):.2f}')

    # Lowest DtG (most dust-poor)
    if len(w_dtg_check) > 0:
        imin_dtg = w_dtg_check[np.argmin(dtg_all)]
        print(f'  Lowest DtG (ColdGas > 10^6):')
        print(f'    DtG = {ColdDust[imin_dtg] / ColdGas[imin_dtg]:.6f},  '
              f'log M* = {np.log10(max(StellarMass[imin_dtg],1)):.2f},  '
              f'log M_gas = {np.log10(ColdGas[imin_dtg]):.2f}')

    # --- 11. Redshift evolution snapshot ---
    print(f'\n  [11] REDSHIFT EVOLUTION (median DtG for ColdGas > 10^4 galaxies)')
    print(f'  {"—"*40}')
    key_snaps = [(63, 0.0), (53, 0.28), (44, 0.76), (37, 1.39), (32, 2.07),
                 (27, 3.06), (23, 3.87), (20, 4.89)]
    print(f'  {"Snap":>5s}  {"z":>5s}  {"N_gal":>7s}  {"N_dust":>7s}  {"med DtG":>10s}  {"med DtM":>10s}  {"rho_dust":>12s}')
    for snapnum, z_approx in key_snaps:
        sn = f'Snap_{snapnum}'
        sm_z = read_all_files(DirName, sn, 'StellarMass')
        cg_z = read_all_files(DirName, sn, 'ColdGas')
        mcg_z = read_all_files(DirName, sn, 'MetalsColdGas')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        hd_z = read_all_files(DirName, sn, 'HotDust')
        ed_z = read_all_files(DirName, sn, 'EjectedDust')
        if cd_z is None:
            print(f'  {snapnum:>5d}  {z_approx:>5.2f}  {"---":>7s}')
            continue
        sm_z = sm_z * 1e10 / Hubble_h
        cg_z = cg_z * 1e10 / Hubble_h
        mcg_z = mcg_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        hd_z = hd_z * 1e10 / Hubble_h
        ed_z = ed_z * 1e10 / Hubble_h

        n_gal_z = len(sm_z)
        wz = np.where((cg_z > 1e4) & (cd_z > 0) & (mcg_z > 0))[0]
        n_dust_z = len(wz)
        med_dtg_z = np.median(cd_z[wz] / cg_z[wz]) if n_dust_z > 5 else np.nan
        med_dtm_z = np.median(cd_z[wz] / mcg_z[wz]) if n_dust_z > 5 else np.nan
        rho_z = (cd_z.sum() + np.maximum(hd_z, 0).sum() + ed_z.sum()) / volume
        print(f'  {snapnum:>5d}  {z_approx:>5.2f}  {n_gal_z:>7d}  {n_dust_z:>7d}  {med_dtg_z:>10.5f}  {med_dtm_z:>10.4f}  {rho_z:>12.3e}')

    print(f'\n{"="*70}\n')

    # ==================================================================
    # Load observational data (used across multiple panels)
    # ==================================================================
    rr = load_remy_ruyer_2014()
    if rr is not None:
        print(f'  Loaded Rémy-Ruyer+2014: {len(rr["logMstar"])} galaxies')
    else:
        print('  WARNING: Could not load Rémy-Ruyer+2014 data')

    # ==================================================================
    # Create 3x2 figure
    # ==================================================================
    fig, axes = plt.subplots(4, 2, figsize=(14, 24))

    # ---- PANEL 1: Cosmic dust density evolution ----
    ax = axes[0, 0]
    zz_list = []
    rho_cold_list = []
    rho_hot_list  = []
    rho_ej_list   = []
    rho_tot_list  = []

    # Sample snapshots (don't need to read every one)
    snap_sample = list(range(15, LastSnap+1, 1))  # z < ~8
    for snapnum in snap_sample:
        sn = f'Snap_{snapnum}'
        cd = read_all_files(DirName, sn, 'ColdDust')
        hd = read_all_files(DirName, sn, 'HotDust')
        ed = read_all_files(DirName, sn, 'EjectedDust')
        if cd is None:
            continue
        cd = cd * 1e10 / Hubble_h
        hd = hd * 1e10 / Hubble_h
        ed = ed * 1e10 / Hubble_h

        zz_list.append(redshifts[snapnum])
        rho_cold_list.append(cd.sum() / volume)
        rho_hot_list.append(np.maximum(hd, 0).sum() / volume)
        rho_ej_list.append(ed.sum() / volume)
        rho_tot_list.append(cd.sum() / volume + np.maximum(hd, 0).sum() / volume + ed.sum() / volume)

    zz_arr = np.array(zz_list)
    ax.semilogy(zz_arr, rho_tot_list,  'w-',  lw=2.5, label='Total')
    ax.semilogy(zz_arr, rho_cold_list, '-',   color='goldenrod', lw=2, label='Cold (ISM)')
    ax.semilogy(zz_arr, rho_hot_list,  '--',  color='firebrick', lw=2, label='Hot (CGM)')
    ax.semilogy(zz_arr, rho_ej_list,   ':',   color='dodgerblue', lw=2, label='Ejected')

    # Observational data: cosmic dust density
    obs_cdd = obs_cosmic_dust_density()
    obs_markers = {'Dunne+11': ('s', 'deepskyblue'), 'Ménard+12': ('D', 'lime')}
    for name, d in obs_cdd.items():
        rho_lin = 10**d['logrho']  # convert to linear for semilogy
        mk, clr = obs_markers.get(name, ('o', 'white'))
        ax.errorbar(d['z'], rho_lin,
                    yerr=[rho_lin - 10**(d['logrho'] - d['errlo']),
                          10**(d['logrho'] + d['errhi']) - rho_lin],
                    fmt=mk, color=clr, ms=6, capsize=3, lw=1.2,
                    label=name, zorder=10)

    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\rho_{\rm dust}\ (M_\odot\,{\rm Mpc}^{-3})$')
    ax.set_xlim(8, 0)
    ax.set_ylim(1e1, 1e8)
    ax.set_title('Cosmic Dust Density Evolution')
    leg = ax.legend(loc='lower right', fontsize=10)
    leg.draw_frame(False)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    # ---- PANEL 2: Dust-to-gas ratio vs metallicity ----
    ax = axes[0, 1]

    w = np.where((ColdGas > 1e4) & (ColdDust > 0) & (MetalsColdGas > 0))[0]
    if len(w) > 0:
        Z_gas = MetalsColdGas[w] / ColdGas[w]
        dtg = ColdDust[w] / ColdGas[w]

        logZ = np.log10(Z_gas / 0.02)  # relative to solar
        logDtG = np.log10(dtg)

        # Scatter
        if len(w) > 10000:
            idx = np.random.choice(len(w), 10000, replace=False)
        else:
            idx = np.arange(len(w))
        ax.scatter(logZ[idx], logDtG[idx], s=1, alpha=0.2, c='goldenrod')

        # Running median
        zbins = np.arange(-2.5, 1.0, 0.2)
        zc = 0.5 * (zbins[:-1] + zbins[1:])
        med_dtg = np.full(len(zc), np.nan)
        p16 = np.full(len(zc), np.nan)
        p84 = np.full(len(zc), np.nan)
        for i in range(len(zc)):
            sel = np.where((logZ >= zbins[i]) & (logZ < zbins[i+1]))[0]
            if len(sel) > 10:
                med_dtg[i] = np.median(logDtG[sel])
                p16[i] = np.percentile(logDtG[sel], 16)
                p84[i] = np.percentile(logDtG[sel], 84)

        good = ~np.isnan(med_dtg)
        ax.plot(zc[good], med_dtg[good], 'w-', lw=2.5, label='SAGE26 median')
        ax.fill_between(zc[good], p16[good], p84[good], color='white', alpha=0.12)

        # Linear reference: DtG = 0.01 * (Z/Zsun) → log(DtG) = -2 + log(Z/Zsun)
        zref = np.linspace(-2.5, 0.5, 50)
        ax.plot(zref, -2.0 + zref, 'c--', lw=1.5, alpha=0.7, label=r'DtG $\propto Z$ (MW norm)')

    # Observational data: Rémy-Ruyer+2014 DtG vs Z
    rr = load_remy_ruyer_2014()
    if rr is not None:
        rr_Z = rr['Z_12logOH'] - 8.69  # convert 12+log(O/H) to log(Z/Zsun)
        rr_Mdust = 10**rr['logMdust']
        rr_Mgas = 10**rr['logMgas']
        ok = np.isfinite(rr['logMgas']) & (rr_Mgas > 0)
        rr_dtg = np.full(len(rr_Mdust), np.nan)
        rr_dtg[ok] = np.log10(rr_Mdust[ok] / rr_Mgas[ok])
        ok2 = np.isfinite(rr_dtg)
        ax.scatter(rr_Z[ok2], rr_dtg[ok2], s=18, marker='o',
                   facecolors='none', edgecolors='deepskyblue', lw=0.7,
                   alpha=0.8, label='Rémy-Ruyer+14', zorder=5)

    ax.set_xlabel(r'$\log_{10}(Z_{\rm gas} / Z_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}^{\rm cold} / M_{\rm cold\,gas})$')
    ax.set_xlim(-2.5, 0.5)
    ax.set_ylim(-6, -0.5)
    ax.set_title('Dust-to-Gas Ratio vs Metallicity')
    leg = ax.legend(loc='lower right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 3: Dust-to-metal ratio vs stellar mass ----
    ax = axes[1, 0]

    w = np.where((StellarMass > 0) & (ColdGas > 1e4) & (ColdDust > 0) & (MetalsColdGas > 0))[0]
    if len(w) > 0:
        mass = np.log10(StellarMass[w])
        fdust = ColdDust[w] / MetalsColdGas[w]
        logfdust = np.log10(np.clip(fdust, 1e-6, None))

        if len(w) > 10000:
            idx = np.random.choice(len(w), 10000, replace=False)
        else:
            idx = np.arange(len(w))
        ax.scatter(mass[idx], logfdust[idx], s=1, alpha=0.2, c='goldenrod')

        # Running median
        mbins = np.arange(7.5, 12.5, 0.25)
        mc = 0.5 * (mbins[:-1] + mbins[1:])
        med_f = np.full(len(mc), np.nan)
        p16_f = np.full(len(mc), np.nan)
        p84_f = np.full(len(mc), np.nan)
        for i in range(len(mc)):
            sel = np.where((mass >= mbins[i]) & (mass < mbins[i+1]))[0]
            if len(sel) > 10:
                med_f[i] = np.median(logfdust[sel])
                p16_f[i] = np.percentile(logfdust[sel], 16)
                p84_f[i] = np.percentile(logfdust[sel], 84)

        good = ~np.isnan(med_f)
        ax.plot(mc[good], med_f[good], 'w-', lw=2.5, label='SAGE26 median')
        ax.fill_between(mc[good], p16_f[good], p84_f[good], color='white', alpha=0.12)

        # MW reference: fdust ~ 0.5
        ax.axhline(np.log10(0.5), color='cyan', ls='--', lw=1.5, alpha=0.7,
                    label=r'$f_{\rm dust} = 0.5$ (MW)')

    # Observational data: Rémy-Ruyer+2014 DtM vs M*
    if rr is not None:
        rr_Z = rr['Z_12logOH'] - 8.69
        rr_Zfrac = 10**rr_Z * 0.02  # mass fraction
        rr_Mdust = 10**rr['logMdust']
        rr_Mgas = 10**rr['logMgas']
        ok = np.isfinite(rr['logMgas']) & (rr_Mgas > 0) & (rr_Zfrac > 0)
        rr_metals = rr_Mgas * rr_Zfrac
        rr_dtm = np.full(len(rr_Mdust), np.nan)
        rr_dtm[ok] = np.log10(rr_Mdust[ok] / rr_metals[ok])
        ok2 = np.isfinite(rr_dtm)
        ax.scatter(rr['logMstar'][ok2], rr_dtm[ok2], s=18, marker='o',
                   facecolors='none', edgecolors='deepskyblue', lw=0.7,
                   alpha=0.8, label='Rémy-Ruyer+14', zorder=5)

    ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}^{\rm cold} / M_{\rm metals}^{\rm cold})$')
    ax.set_xlim(8, 12)
    ax.set_ylim(-4, 0.5)
    ax.set_title('Dust-to-Metal Ratio vs Stellar Mass')
    leg = ax.legend(loc='lower right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 4: Dust mass function at z=0 ----
    ax = axes[1, 1]
    binwidth = 0.2

    w = np.where(TotalDust > 0)[0]
    if len(w) > 0:
        dustmass = np.log10(TotalDust[w])
        mi, ma = 4.0, 10.0
        NB = int((ma - mi) / binwidth)
        counts, binedges = np.histogram(dustmass, range=(mi, ma), bins=NB)
        xax = binedges[:-1] + 0.5 * binwidth
        phi = counts / volume / binwidth
        phi[phi > 0] = np.log10(phi[phi > 0])
        phi[phi == 0] = np.nan
        ax.plot(xax, phi, 'w-', lw=2.5, label='Total')

    # Cold dust only
    w = np.where(ColdDust > 0)[0]
    if len(w) > 0:
        dustmass_c = np.log10(ColdDust[w])
        counts_c, _ = np.histogram(dustmass_c, range=(mi, ma), bins=NB)
        phi_c = counts_c / volume / binwidth
        phi_c[phi_c > 0] = np.log10(phi_c[phi_c > 0])
        phi_c[phi_c == 0] = np.nan
        ax.plot(xax, phi_c, '-', color='goldenrod', lw=2, label='Cold (ISM)')

    # Hot dust only
    w = np.where(HotDust > 0)[0]
    if len(w) > 0:
        dustmass_h = np.log10(HotDust[w])
        counts_h, _ = np.histogram(dustmass_h, range=(mi, ma), bins=NB)
        phi_h = counts_h / volume / binwidth
        phi_h[phi_h > 0] = np.log10(phi_h[phi_h > 0])
        phi_h[phi_h == 0] = np.nan
        ax.plot(xax, phi_h, '--', color='firebrick', lw=2, label='Hot (CGM)')

    # Observational data: DMF at z=0
    obs_dmf = obs_dmf_z0()
    dmf_markers = {'Vlahakis+05': ('s', 'deepskyblue'),
                   'Clemens+13': ('^', 'lime'),
                   'Dunne+11':   ('D', 'salmon')}
    for name, d in obs_dmf.items():
        mk, clr = dmf_markers.get(name, ('o', 'white'))
        ax.errorbar(d['logMd'], d['logphi'],
                    yerr=[d['errlo'], d['errhi']],
                    fmt=mk, color=clr, ms=5, capsize=2, lw=1,
                    label=name, zorder=10)

    ax.set_xlabel(r'$\log_{10}\, M_{\rm dust}\ (M_\odot)$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ (\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$')
    ax.set_xlim(4, 10)
    ax.set_ylim(-6, -0.5)
    ax.set_title('Dust Mass Function (z = 0)')
    leg = ax.legend(loc='upper right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 5: Dust reservoir breakdown vs stellar mass ----
    ax = axes[2, 0]

    w = np.where(StellarMass > 0)[0]
    if len(w) > 0:
        mass = np.log10(StellarMass[w])
        mbins = np.arange(7.5, 12.5, 0.25)
        mc = 0.5 * (mbins[:-1] + mbins[1:])

        med_cold = np.full(len(mc), np.nan)
        med_hot  = np.full(len(mc), np.nan)
        med_ej   = np.full(len(mc), np.nan)

        for i in range(len(mc)):
            sel = np.where((mass >= mbins[i]) & (mass < mbins[i+1]))[0]
            if len(sel) > 10:
                cd_sel = np.maximum(ColdDust[w[sel]], 0)
                hd_sel = np.maximum(HotDust[w[sel]], 0)
                ed_sel = np.maximum(EjectedDust[w[sel]], 0)
                td_sel = cd_sel + hd_sel + ed_sel
                ok = td_sel > 0
                if ok.sum() > 5:
                    med_cold[i] = np.median(cd_sel[ok] / td_sel[ok])
                    med_hot[i]  = np.median(hd_sel[ok] / td_sel[ok])
                    med_ej[i]   = np.median(ed_sel[ok] / td_sel[ok])

        good = ~np.isnan(med_cold)
        ax.plot(mc[good], med_cold[good], '-',  color='goldenrod', lw=2.5, label='Cold (ISM)')
        ax.plot(mc[good], med_hot[good],  '--', color='firebrick', lw=2.5, label='Hot (CGM)')
        ax.plot(mc[good], med_ej[good],   ':',  color='dodgerblue', lw=2.5, label='Ejected')

    ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
    ax.set_ylabel('Median dust fraction in reservoir')
    ax.set_xlim(8, 12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Dust Reservoir Breakdown')
    leg = ax.legend(loc='center right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 6: Dust-to-gas ratio vs stellar mass ----
    ax = axes[2, 1]

    w = np.where((StellarMass > 0) & (ColdGas > 1e4) & (ColdDust > 0))[0]
    if len(w) > 0:
        mass = np.log10(StellarMass[w])
        dtg = np.log10(ColdDust[w] / ColdGas[w])

        if len(w) > 10000:
            idx = np.random.choice(len(w), 10000, replace=False)
        else:
            idx = np.arange(len(w))
        ax.scatter(mass[idx], dtg[idx], s=1, alpha=0.2, c='goldenrod')

        # Running median
        mbins = np.arange(7.5, 12.5, 0.25)
        mc = 0.5 * (mbins[:-1] + mbins[1:])
        med_dtg = np.full(len(mc), np.nan)
        p16_dtg = np.full(len(mc), np.nan)
        p84_dtg = np.full(len(mc), np.nan)
        for i in range(len(mc)):
            sel = np.where((mass >= mbins[i]) & (mass < mbins[i+1]))[0]
            if len(sel) > 10:
                med_dtg[i] = np.median(dtg[sel])
                p16_dtg[i] = np.percentile(dtg[sel], 16)
                p84_dtg[i] = np.percentile(dtg[sel], 84)

        good = ~np.isnan(med_dtg)
        ax.plot(mc[good], med_dtg[good], 'w-', lw=2.5, label='SAGE26 median')
        ax.fill_between(mc[good], p16_dtg[good], p84_dtg[good],
                         color='white', alpha=0.12, label=r'16–84th pctl')

        # MW reference
        ax.axhline(np.log10(0.01), color='cyan', ls='--', lw=1.5, label='MW DtG (Draine 03)')

    # Observational data: Rémy-Ruyer+2014 DtG vs M*
    if rr is not None:
        rr_Mdust = 10**rr['logMdust']
        rr_Mgas = 10**rr['logMgas']
        ok = np.isfinite(rr['logMgas']) & (rr_Mgas > 0)
        rr_dtg_ms = np.full(len(rr_Mdust), np.nan)
        rr_dtg_ms[ok] = np.log10(rr_Mdust[ok] / rr_Mgas[ok])
        ok2 = np.isfinite(rr_dtg_ms)
        ax.scatter(rr['logMstar'][ok2], rr_dtg_ms[ok2], s=18, marker='o',
                   facecolors='none', edgecolors='deepskyblue', lw=0.7,
                   alpha=0.8, label='Rémy-Ruyer+14', zorder=5)

    ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}^{\rm cold} / M_{\rm cold\,gas})$')
    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -0.5)
    ax.set_title('Dust-to-Gas Ratio vs Stellar Mass')
    leg = ax.legend(loc='lower right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 7: Specific dust mass (M_dust/M_star) vs stellar mass ----
    ax = axes[3, 0]

    w = np.where((StellarMass > 0) & (ColdDust > 0))[0]
    if len(w) > 0:
        mass = np.log10(StellarMass[w])
        sdm = np.log10(ColdDust[w] / StellarMass[w])

        if len(w) > 10000:
            idx = np.random.choice(len(w), 10000, replace=False)
        else:
            idx = np.arange(len(w))
        ax.scatter(mass[idx], sdm[idx], s=1, alpha=0.2, c='goldenrod')

        # Running median
        mbins = np.arange(7.5, 12.5, 0.25)
        mc = 0.5 * (mbins[:-1] + mbins[1:])
        med_sdm = np.full(len(mc), np.nan)
        p16_sdm = np.full(len(mc), np.nan)
        p84_sdm = np.full(len(mc), np.nan)
        for i in range(len(mc)):
            sel = np.where((mass >= mbins[i]) & (mass < mbins[i+1]))[0]
            if len(sel) > 10:
                med_sdm[i] = np.median(sdm[sel])
                p16_sdm[i] = np.percentile(sdm[sel], 16)
                p84_sdm[i] = np.percentile(sdm[sel], 84)

        good = ~np.isnan(med_sdm)
        ax.plot(mc[good], med_sdm[good], 'w-', lw=2.5, label='SAGE26 median')
        ax.fill_between(mc[good], p16_sdm[good], p84_sdm[good],
                         color='white', alpha=0.12, label=r'16–84th pctl')

        # MW reference: M_dust ~ 5e7, M_star ~ 5e10 → log(ratio) ~ -3.0
        ax.axhline(-3.0, color='cyan', ls='--', lw=1.5, label=r'MW $M_d/M_\star$ ~ $10^{-3}$')

    # Observational data: Rémy-Ruyer+2014 Mdust/Mstar vs M*
    if rr is not None:
        rr_sdm = rr['logMdust'] - rr['logMstar']
        ok = np.isfinite(rr_sdm)
        ax.scatter(rr['logMstar'][ok], rr_sdm[ok], s=18, marker='o',
                   facecolors='none', edgecolors='deepskyblue', lw=0.7,
                   alpha=0.8, label='Rémy-Ruyer+14', zorder=5)

    # DustPedia / Nersesian+2019
    dp = obs_dustpedia_nersesian2019()
    ax.errorbar(dp['logMstar'], dp['logMdust'] - dp['logMstar'],
                xerr=dp['logMstar_err'], yerr=dp['logMdust_err'],
                fmt='s', color='lime', ms=5, capsize=2, lw=1,
                label='Nersesian+19', zorder=10)

    ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}^{\rm cold} / M_\star)$')
    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -0.5)
    ax.set_title('Specific Dust Mass vs Stellar Mass')
    leg = ax.legend(loc='upper right', fontsize=10)
    leg.draw_frame(False)

    # ---- PANEL 8: DtG evolution at fixed mass bins ----
    ax = axes[3, 1]

    mass_bins_evol = [(9.0, 9.5, 'mediumpurple', r'$9.0 < \log M_\star < 9.5$'),
                      (9.5, 10.0, 'goldenrod',    r'$9.5 < \log M_\star < 10.0$'),
                      (10.0, 10.5, 'limegreen',   r'$10.0 < \log M_\star < 10.5$'),
                      (10.5, 11.0, 'tomato',      r'$10.5 < \log M_\star < 11.0$')]

    for mlo, mhi, colour, mlabel in mass_bins_evol:
        zz_evol = []
        dtg_evol = []
        for snapnum in snap_sample:
            sn = f'Snap_{snapnum}'
            sm_e = read_all_files(DirName, sn, 'StellarMass')
            cg_e = read_all_files(DirName, sn, 'ColdGas')
            cd_e = read_all_files(DirName, sn, 'ColdDust')
            if sm_e is None or cd_e is None:
                continue
            sm_e = sm_e * 1e10 / Hubble_h
            cg_e = cg_e * 1e10 / Hubble_h
            cd_e = cd_e * 1e10 / Hubble_h

            with np.errstate(divide='ignore', invalid='ignore'):
                logsm_e = np.where(sm_e > 0, np.log10(sm_e), -99.0)
            we = np.where((logsm_e >= mlo) & (logsm_e < mhi) &
                          (cg_e > 1e4) & (cd_e > 0))[0]
            if len(we) > 10:
                zz_evol.append(redshifts[snapnum])
                dtg_evol.append(np.log10(np.median(cd_e[we] / cg_e[we])))

        if len(zz_evol) > 2:
            ax.plot(zz_evol, dtg_evol, '-', color=colour, lw=2, label=mlabel)

    ax.axhline(np.log10(0.01), color='cyan', ls='--', lw=1, alpha=0.5, label='MW DtG')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$\log_{10}$ median DtG')
    ax.set_xlim(6, 0)
    ax.set_ylim(-5, -0.5)
    ax.set_title('DtG Evolution by Stellar Mass Bin')
    leg = ax.legend(loc='lower right', fontsize=9)
    leg.draw_frame(False)

    # ==================================================================
    # Compare with second run if provided
    # ==================================================================
    if CompareDir and os.path.exists(os.path.join(CompareDir, 'model_0.hdf5')):
        print(f'\n  Comparing with: {CompareDir}')
        snap_str2 = f'Snap_{LastSnap}'

        SM2   = read_all_files(CompareDir, snap_str2, 'StellarMass') * 1e10 / Hubble_h
        CG2   = read_all_files(CompareDir, snap_str2, 'ColdGas')     * 1e10 / Hubble_h
        MCG2  = read_all_files(CompareDir, snap_str2, 'MetalsColdGas') * 1e10 / Hubble_h
        CD2   = read_all_files(CompareDir, snap_str2, 'ColdDust')    * 1e10 / Hubble_h
        HD2   = read_all_files(CompareDir, snap_str2, 'HotDust')     * 1e10 / Hubble_h
        ED2   = read_all_files(CompareDir, snap_str2, 'EjectedDust') * 1e10 / Hubble_h
        TD2   = CD2 + HD2 + ED2

        if CD2 is not None:
            # Overlay comparison on Panel 2 (DtG vs Z)
            ax = axes[0, 1]
            w2 = np.where((CG2 > 1e4) & (CD2 > 0) & (MCG2 > 0))[0]
            if len(w2) > 0:
                Z2 = MCG2[w2] / CG2[w2]
                dtg2 = CD2[w2] / CG2[w2]
                logZ2 = np.log10(Z2 / 0.02)
                logDtG2 = np.log10(dtg2)

                zbins = np.arange(-2.5, 1.0, 0.2)
                zc = 0.5 * (zbins[:-1] + zbins[1:])
                med2 = np.full(len(zc), np.nan)
                for i in range(len(zc)):
                    sel = np.where((logZ2 >= zbins[i]) & (logZ2 < zbins[i+1]))[0]
                    if len(sel) > 10:
                        med2[i] = np.median(logDtG2[sel])
                good = ~np.isnan(med2)
                ax.plot(zc[good], med2[good], 'r-', lw=2, label='Compare median')
                ax.legend(loc='lower right', fontsize=9).draw_frame(False)

            # Overlay on Panel 6 (DtG vs Mstar)
            ax = axes[2, 1]
            w2 = np.where((SM2 > 0) & (CG2 > 1e4) & (CD2 > 0))[0]
            if len(w2) > 0:
                mass2 = np.log10(SM2[w2])
                dtg2 = np.log10(CD2[w2] / CG2[w2])
                mbins = np.arange(7.5, 12.5, 0.25)
                mc = 0.5 * (mbins[:-1] + mbins[1:])
                med2 = np.full(len(mc), np.nan)
                for i in range(len(mc)):
                    sel = np.where((mass2 >= mbins[i]) & (mass2 < mbins[i+1]))[0]
                    if len(sel) > 10:
                        med2[i] = np.median(dtg2[sel])
                good = ~np.isnan(med2)
                ax.plot(mc[good], med2[good], 'r-', lw=2, label='Compare median')
                ax.legend(loc='lower right', fontsize=9).draw_frame(False)

            # Overlay on Panel 4 (DMF)
            ax = axes[1, 1]
            w2 = np.where(TD2 > 0)[0]
            if len(w2) > 0:
                dm2 = np.log10(TD2[w2])
                vol2 = volume
                counts2, _ = np.histogram(dm2, range=(4.0, 10.0), bins=NB)
                phi2 = counts2 / vol2 / binwidth
                phi2[phi2 > 0] = np.log10(phi2[phi2 > 0])
                phi2[phi2 == 0] = np.nan
                ax.plot(xax, phi2, 'r-', lw=2, label='Compare (total)')
                ax.legend(loc='upper right', fontsize=9).draw_frame(False)

            # Cosmic density evolution comparison
            ax = axes[0, 0]
            rho_compare = []
            zz_compare = []
            for snapnum in snap_sample:
                sn = f'Snap_{snapnum}'
                cd2 = read_all_files(CompareDir, sn, 'ColdDust')
                hd2 = read_all_files(CompareDir, sn, 'HotDust')
                ed2 = read_all_files(CompareDir, sn, 'EjectedDust')
                if cd2 is not None:
                    cd2 = cd2 * 1e10 / Hubble_h
                    hd2 = hd2 * 1e10 / Hubble_h
                    ed2 = ed2 * 1e10 / Hubble_h
                    zz_compare.append(redshifts[snapnum])
                    rho_compare.append((cd2.sum() + np.maximum(hd2, 0).sum() + ed2.sum()) / volume)
            if len(rho_compare) > 0:
                ax.semilogy(zz_compare, rho_compare, 'r-', lw=2, label='Compare (total)')
                ax.legend(loc='lower right', fontsize=9).draw_frame(False)

    # ==================================================================
    # Save
    # ==================================================================
    plt.tight_layout(h_pad=3.0)
    outfile = os.path.join(OutputDir, 'DustDiagnostics' + OutputFormat)
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    print(f'  Saved 8-panel figure to: {outfile}\n')
    plt.close()

    # ==================================================================
    # NEW FIGURE: Dust Mass vs Stellar Mass Evolution (z=0 to z=7)
    # ==================================================================
    print(f'  Creating Dust Mass vs Stellar Mass evolution figure...')
    
    # Define redshift targets and corresponding snapshots
    # Millennium snapshots: z=0 (63), z~1 (37), z~2 (32), z~3 (27), z~4 (23), z~5 (20), z~6 (17), z~7 (15)
    z_panels = [
        (63, 0.00, 'z = 0'),
        (37, 1.39, 'z = 1'),
        (32, 2.07, 'z = 2'),
        (27, 3.06, 'z = 3'),
        (23, 3.87, 'z = 4'),
        (20, 4.89, 'z = 5'),
        (17, 6.20, 'z = 6'),
        (15, 7.27, 'z = 7'),
    ]
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10))
    axes2 = axes2.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes2[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        sm_z = read_all_files(DirName, sn, 'StellarMass')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        hd_z = read_all_files(DirName, sn, 'HotDust')
        ed_z = read_all_files(DirName, sn, 'EjectedDust')
        cgm_z = read_all_files(DirName, sn, 'CGMDust')
        
        if sm_z is None or cd_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(3, 10)
            continue
        
        # Convert to physical units
        sm_z = sm_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        hd_z = hd_z * 1e10 / Hubble_h
        ed_z = ed_z * 1e10 / Hubble_h
        if cgm_z is not None:
            cgm_z = cgm_z * 1e10 / Hubble_h
            total_dust_z = cd_z + np.maximum(hd_z, 0) + ed_z + np.maximum(cgm_z, 0)
        else:
            total_dust_z = cd_z + np.maximum(hd_z, 0) + ed_z
        
        # Select galaxies with positive stellar and dust mass
        w = np.where((sm_z > 0) & (total_dust_z > 0))[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(3, 10)
            continue
        
        log_mstar = np.log10(sm_z[w])
        log_mdust = np.log10(total_dust_z[w])
        
        # Create 2D histogram (heatmap)
        xbins = np.linspace(7, 12, 60)
        ybins = np.linspace(3, 10, 60)
        
        H, xedges, yedges = np.histogram2d(log_mstar, log_mdust, bins=[xbins, ybins])
        
        # Use raw counts for density display
        H_plot = H.T
        
        # Create heatmap with custom colormap (dark background compatible)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        cmap.set_under(color='black')
        
        im = ax.imshow(H_plot, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, vmin=1, interpolation='gaussian')
        
        # Calculate and plot median line
        mstar_bins = np.arange(7.5, 12.0, 0.25)
        mstar_centers = 0.5 * (mstar_bins[:-1] + mstar_bins[1:])
        median_mdust = np.full(len(mstar_centers), np.nan)
        p16_mdust = np.full(len(mstar_centers), np.nan)
        p84_mdust = np.full(len(mstar_centers), np.nan)
        
        for i in range(len(mstar_centers)):
            sel = np.where((log_mstar >= mstar_bins[i]) & (log_mstar < mstar_bins[i+1]))[0]
            if len(sel) > 10:
                median_mdust[i] = np.median(log_mdust[sel])
                p16_mdust[i] = np.percentile(log_mdust[sel], 16)
                p84_mdust[i] = np.percentile(log_mdust[sel], 84)
        
        good = ~np.isnan(median_mdust)
        ax.plot(mstar_centers[good], median_mdust[good], 'k-', lw=3, label='Median')
        ax.plot(mstar_centers[good], median_mdust[good], 'w-', lw=1.5)
        
        # Labels and styling
        ax.set_xlim(7, 12)
        ax.set_ylim(3, 10)
        ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
        ax.set_ylabel(r'$\log_{10}\, M_{\rm dust}\ (M_\odot)$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig2.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$N_{\rm gal}$', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    fig2.suptitle('Dust Mass vs Stellar Mass Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    outfile2 = os.path.join(OutputDir, 'DustMass_StellarMass_Evolution' + OutputFormat)
    fig2.savefig(outfile2, facecolor=fig2.get_facecolor())
    print(f'  Saved dust-stellar mass evolution figure to: {outfile2}\n')
    plt.close(fig2)

    # ==================================================================
    # NEW FIGURE: Dust Components vs Stellar Mass (line plot version)
    # ==================================================================
    print(f'  Creating Dust Components vs Stellar Mass evolution figure...')
    
    fig2b, axes2b = plt.subplots(2, 4, figsize=(18, 10))
    axes2b = axes2b.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes2b[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        sm_z = read_all_files(DirName, sn, 'StellarMass')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        hd_z = read_all_files(DirName, sn, 'HotDust')
        ed_z = read_all_files(DirName, sn, 'EjectedDust')
        cgm_z = read_all_files(DirName, sn, 'CGMDust')
        
        if sm_z is None or cd_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(3, 10)
            continue
        
        # Convert to physical units
        sm_z = sm_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        hd_z = hd_z * 1e10 / Hubble_h
        ed_z = ed_z * 1e10 / Hubble_h
        if cgm_z is not None:
            cgm_z = cgm_z * 1e10 / Hubble_h
            total_dust_z = cd_z + np.maximum(hd_z, 0) + ed_z + np.maximum(cgm_z, 0)
        else:
            total_dust_z = cd_z + np.maximum(hd_z, 0) + ed_z
        
        # Select galaxies with positive stellar mass
        w = np.where(sm_z > 0)[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(3, 10)
            continue
        
        log_mstar = np.log10(sm_z[w])
        
        # Define stellar mass bins
        mstar_bins = np.arange(7.5, 12.0, 0.25)
        mstar_centers = 0.5 * (mstar_bins[:-1] + mstar_bins[1:])
        
        # Define dust component arrays to iterate over
        dust_components = [
            (total_dust_z[w], 'Total', 'white', '-', 2.5),
            (cd_z[w], 'Cold', 'cyan', '-', 2.0),
            (np.maximum(hd_z[w], 0), 'Hot', 'red', '--', 2.0),
            (ed_z[w], 'Ejected', 'orange', ':', 2.0),
        ]
        
        for dust_arr, label, color, ls, lw in dust_components:
            median_mdust = np.full(len(mstar_centers), np.nan)
            p16_mdust = np.full(len(mstar_centers), np.nan)
            p84_mdust = np.full(len(mstar_centers), np.nan)
            
            for i in range(len(mstar_centers)):
                sel = np.where((log_mstar >= mstar_bins[i]) & (log_mstar < mstar_bins[i+1]) & (dust_arr > 0))[0]
                if len(sel) > 10:
                    log_dust = np.log10(dust_arr[sel])
                    median_mdust[i] = np.median(log_dust)
                    p16_mdust[i] = np.percentile(log_dust, 16)
                    p84_mdust[i] = np.percentile(log_dust, 84)
            
            good = ~np.isnan(median_mdust)
            if np.sum(good) > 2:
                ax.plot(mstar_centers[good], median_mdust[good], color=color, ls=ls, lw=lw, label=label)
                # Add shaded region for Total only
                if label == 'Total':
                    ax.fill_between(mstar_centers[good], p16_mdust[good], p84_mdust[good], 
                                   color=color, alpha=0.15)
        
        # Labels and styling
        ax.set_xlim(7, 12)
        ax.set_ylim(3, 10)
        ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
        ax.set_ylabel(r'$\log_{10}\, M_{\rm dust}\ (M_\odot)$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Add legend to first panel only
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
    
    fig2b.suptitle('Dust Components vs Stellar Mass Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    
    outfile2b = os.path.join(OutputDir, 'DustComponents_StellarMass_Evolution' + OutputFormat)
    fig2b.savefig(outfile2b, facecolor=fig2b.get_facecolor())
    print(f'  Saved dust components evolution figure to: {outfile2b}\n')
    plt.close(fig2b)

    # ==================================================================
    # NEW FIGURE: Dust-to-Gas Ratio vs Stellar Mass Evolution (z=0 to z=7)
    # ==================================================================
    print(f'  Creating Dust Fraction vs Stellar Mass evolution figure...')
    
    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 10))
    axes3 = axes3.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes3[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        sm_z = read_all_files(DirName, sn, 'StellarMass')
        cg_z = read_all_files(DirName, sn, 'ColdGas')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        
        if sm_z is None or cd_z is None or cg_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(-6, 0)
            continue
        
        # Convert to physical units
        sm_z = sm_z * 1e10 / Hubble_h
        cg_z = cg_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        
        # Select galaxies with positive stellar mass, cold gas, and cold dust
        w = np.where((sm_z > 0) & (cg_z > 1e4) & (cd_z > 0))[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(-6, 0)
            continue
        
        log_mstar = np.log10(sm_z[w])
        log_dtg = np.log10(cd_z[w] / cg_z[w])
        
        # Create 2D histogram (heatmap)
        xbins = np.linspace(7, 12, 60)
        ybins = np.linspace(-6, 0, 60)
        
        H, xedges, yedges = np.histogram2d(log_mstar, log_dtg, bins=[xbins, ybins])
        
        # Use raw counts for density display
        H_plot = H.T
        
        # Create heatmap with custom colormap (dark background compatible)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        cmap.set_under(color='black')
        
        im3 = ax.imshow(H_plot, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, vmin=1, interpolation='gaussian')
        
        # Calculate and plot median line
        mstar_bins = np.arange(7.5, 12.0, 0.25)
        mstar_centers = 0.5 * (mstar_bins[:-1] + mstar_bins[1:])
        median_dtg = np.full(len(mstar_centers), np.nan)
        
        for i in range(len(mstar_centers)):
            sel = np.where((log_mstar >= mstar_bins[i]) & (log_mstar < mstar_bins[i+1]))[0]
            if len(sel) > 10:
                median_dtg[i] = np.median(log_dtg[sel])
        
        good = ~np.isnan(median_dtg)
        ax.plot(mstar_centers[good], median_dtg[good], 'k-', lw=3, label='Median')
        ax.plot(mstar_centers[good], median_dtg[good], 'w-', lw=1.5)
        
        # MW reference line
        ax.axhline(np.log10(0.01), color='cyan', ls='--', lw=1.5, alpha=0.7)
        
        # Labels and styling
        ax.set_xlim(7, 12)
        ax.set_ylim(-6, 0)
        ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_{\rm gas})$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax3 = fig3.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar3 = fig3.colorbar(im3, cax=cbar_ax3)
    cbar3.set_label(r'$N_{\rm gal}$', color='white')
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')
    
    fig3.suptitle('Dust-to-Gas Ratio vs Stellar Mass Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    outfile3 = os.path.join(OutputDir, 'DustFraction_StellarMass_Evolution' + OutputFormat)
    fig3.savefig(outfile3, facecolor=fig3.get_facecolor())
    print(f'  Saved dust fraction evolution figure to: {outfile3}\n')
    plt.close(fig3)

    # ==================================================================
    # NEW FIGURE: Dust-to-Gas Ratio vs Metallicity Evolution (z=0 to z=7)
    # ==================================================================
    print(f'  Creating Dust Fraction vs Metallicity evolution figure...')
    
    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 10))
    axes4 = axes4.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes4[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        cg_z = read_all_files(DirName, sn, 'ColdGas')
        mcg_z = read_all_files(DirName, sn, 'MetalsColdGas')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        
        if cg_z is None or cd_z is None or mcg_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(6, 10)
            ax.set_ylim(-6, 0)
            continue
        
        # Convert to physical units
        cg_z = cg_z * 1e10 / Hubble_h
        mcg_z = mcg_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        
        # Select galaxies with positive cold gas, metals, and dust
        w = np.where((cg_z > 1e4) & (mcg_z > 0) & (cd_z > 0))[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(6, 10)
            ax.set_ylim(-6, 0)
            continue
        
        # Calculate metallicity as 12 + log(O/H)
        # Z_gas = MetalsColdGas / ColdGas (mass fraction)
        # 12 + log(O/H) = 8.69 + log(Z_gas / 0.02)  [solar = 8.69]
        Z_gas = mcg_z[w] / cg_z[w]
        log_OH = 8.69 + np.log10(Z_gas / 0.02)
        log_dtg = np.log10(cd_z[w] / cg_z[w])
        
        # Create 2D histogram (heatmap)
        xbins = np.linspace(6, 10, 60)
        ybins = np.linspace(-6, 0, 60)
        
        H, xedges, yedges = np.histogram2d(log_OH, log_dtg, bins=[xbins, ybins])
        
        # Use raw counts for density display
        H_plot = H.T
        
        # Create heatmap with custom colormap (dark background compatible)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        cmap.set_under(color='black')
        
        im4 = ax.imshow(H_plot, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, vmin=1, interpolation='gaussian')
        
        # Calculate and plot median line
        oh_bins = np.arange(6.5, 10.0, 0.25)
        oh_centers = 0.5 * (oh_bins[:-1] + oh_bins[1:])
        median_dtg = np.full(len(oh_centers), np.nan)
        
        for i in range(len(oh_centers)):
            sel = np.where((log_OH >= oh_bins[i]) & (log_OH < oh_bins[i+1]))[0]
            if len(sel) > 10:
                median_dtg[i] = np.median(log_dtg[sel])
        
        good = ~np.isnan(median_dtg)
        ax.plot(oh_centers[good], median_dtg[good], 'k-', lw=3, label='Median')
        ax.plot(oh_centers[good], median_dtg[good], 'w-', lw=1.5)
        
        # MW reference line and solar metallicity
        ax.axhline(np.log10(0.01), color='cyan', ls='--', lw=1.5, alpha=0.7)
        ax.axvline(8.69, color='cyan', ls=':', lw=1.5, alpha=0.7)  # Solar 12+log(O/H)
        
        # Labels and styling
        ax.set_xlim(6, 10)
        ax.set_ylim(-6, 0)
        ax.set_xlabel(r'$12 + \log({\rm O/H})$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_{\rm gas})$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax4 = fig4.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar4 = fig4.colorbar(im4, cax=cbar_ax4)
    cbar4.set_label(r'$N_{\rm gal}$', color='white')
    cbar4.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar4.ax.axes, 'yticklabels'), color='white')
    
    fig4.suptitle('Dust-to-Gas Ratio vs Metallicity Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    outfile4 = os.path.join(OutputDir, 'DustFraction_Metallicity_Evolution' + OutputFormat)
    fig4.savefig(outfile4, facecolor=fig4.get_facecolor())
    print(f'  Saved dust fraction vs metallicity figure to: {outfile4}\n')
    plt.close(fig4)

    # ==================================================================
    # NEW FIGURE: Dust-to-Metal Ratio vs Stellar Mass Evolution (z=0 to z=7)
    # ==================================================================
    print(f'  Creating Dust-to-Metal vs Stellar Mass evolution figure...')
    
    fig5, axes5 = plt.subplots(2, 4, figsize=(18, 10))
    axes5 = axes5.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes5[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        sm_z = read_all_files(DirName, sn, 'StellarMass')
        cg_z = read_all_files(DirName, sn, 'ColdGas')
        mcg_z = read_all_files(DirName, sn, 'MetalsColdGas')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        
        if sm_z is None or cd_z is None or mcg_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(-1, 0.5)
            continue
        
        # Convert to physical units
        sm_z = sm_z * 1e10 / Hubble_h
        cg_z = cg_z * 1e10 / Hubble_h
        mcg_z = mcg_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        
        # Select galaxies with positive stellar mass, metals, and dust
        w = np.where((sm_z > 0) & (cg_z > 1e4) & (mcg_z > 0) & (cd_z > 0))[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(7, 12)
            ax.set_ylim(-1, 0.5)
            continue
        
        log_mstar = np.log10(sm_z[w])
        log_dtm = np.log10(cd_z[w] / mcg_z[w])
        
        # Create 2D histogram (heatmap)
        xbins = np.linspace(7, 12, 60)
        ybins = np.linspace(-4, 0.5, 60)
        
        H, xedges, yedges = np.histogram2d(log_mstar, log_dtm, bins=[xbins, ybins])
        
        # Use raw counts for density display
        H_plot = H.T
        
        # Create heatmap with custom colormap (dark background compatible)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        cmap.set_under(color='black')
        
        im5 = ax.imshow(H_plot, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, vmin=1, interpolation='gaussian')
        
        # Calculate and plot median line
        mstar_bins = np.arange(7.5, 12.0, 0.25)
        mstar_centers = 0.5 * (mstar_bins[:-1] + mstar_bins[1:])
        median_dtm = np.full(len(mstar_centers), np.nan)
        
        for i in range(len(mstar_centers)):
            sel = np.where((log_mstar >= mstar_bins[i]) & (log_mstar < mstar_bins[i+1]))[0]
            if len(sel) > 10:
                median_dtm[i] = np.median(log_dtm[sel])
        
        good = ~np.isnan(median_dtm)
        ax.plot(mstar_centers[good], median_dtm[good], 'k-', lw=3, label='Median')
        ax.plot(mstar_centers[good], median_dtm[good], 'w-', lw=1.5)
        
        # MW reference line (DtM ~ 0.5)
        ax.axhline(np.log10(0.5), color='cyan', ls='--', lw=1.5, alpha=0.7)
        
        # Labels and styling
        ax.set_xlim(7, 12)
        ax.set_ylim(-1, 0.5)
        ax.set_xlabel(r'$\log_{10}\, M_\star\ (M_\odot)$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_{\rm metals})$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax5 = fig5.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar5 = fig5.colorbar(im5, cax=cbar_ax5)
    cbar5.set_label(r'$N_{\rm gal}$', color='white')
    cbar5.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar5.ax.axes, 'yticklabels'), color='white')
    
    fig5.suptitle('Dust-to-Metal Ratio vs Stellar Mass Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    outfile5 = os.path.join(OutputDir, 'DustToMetal_StellarMass_Evolution' + OutputFormat)
    fig5.savefig(outfile5, facecolor=fig5.get_facecolor())
    print(f'  Saved dust-to-metal vs stellar mass figure to: {outfile5}\n')
    plt.close(fig5)

    # ==================================================================
    # NEW FIGURE: Dust-to-Metal Ratio vs Metallicity Evolution (z=0 to z=7)
    # ==================================================================
    print(f'  Creating Dust-to-Metal vs Metallicity evolution figure...')
    
    fig6, axes6 = plt.subplots(2, 4, figsize=(18, 10))
    axes6 = axes6.flatten()
    
    for idx, (snapnum, z_val, z_label) in enumerate(z_panels):
        ax = axes6[idx]
        sn = f'Snap_{snapnum}'
        
        # Load data for this snapshot
        cg_z = read_all_files(DirName, sn, 'ColdGas')
        mcg_z = read_all_files(DirName, sn, 'MetalsColdGas')
        cd_z = read_all_files(DirName, sn, 'ColdDust')
        
        if cg_z is None or cd_z is None or mcg_z is None:
            ax.text(0.5, 0.5, f'{z_label}\nNo data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(6, 10)
            ax.set_ylim(-1, 0.5)
            continue
        
        # Convert to physical units
        cg_z = cg_z * 1e10 / Hubble_h
        mcg_z = mcg_z * 1e10 / Hubble_h
        cd_z = cd_z * 1e10 / Hubble_h
        
        # Select galaxies with positive cold gas, metals, and dust
        w = np.where((cg_z > 1e4) & (mcg_z > 0) & (cd_z > 0))[0]
        
        if len(w) < 100:
            ax.text(0.5, 0.5, f'{z_label}\nToo few galaxies', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(6, 10)
            ax.set_ylim(-1, 0.5)
            continue
        
        # Calculate metallicity as 12 + log(O/H)
        Z_gas = mcg_z[w] / cg_z[w]
        log_OH = 8.69 + np.log10(Z_gas / 0.02)
        log_dtm = np.log10(cd_z[w] / mcg_z[w])
        
        # Create 2D histogram (heatmap)
        xbins = np.linspace(6, 10, 60)
        ybins = np.linspace(-4, 0.5, 60)
        
        H, xedges, yedges = np.histogram2d(log_OH, log_dtm, bins=[xbins, ybins])
        
        # Use raw counts for density display
        H_plot = H.T
        
        # Create heatmap with custom colormap (dark background compatible)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')
        cmap.set_under(color='black')
        
        im6 = ax.imshow(H_plot, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap, vmin=1, interpolation='gaussian')
        
        # Calculate and plot median line
        oh_bins = np.arange(6.5, 10.0, 0.25)
        oh_centers = 0.5 * (oh_bins[:-1] + oh_bins[1:])
        median_dtm = np.full(len(oh_centers), np.nan)
        
        for i in range(len(oh_centers)):
            sel = np.where((log_OH >= oh_bins[i]) & (log_OH < oh_bins[i+1]))[0]
            if len(sel) > 10:
                median_dtm[i] = np.median(log_dtm[sel])
        
        good = ~np.isnan(median_dtm)
        ax.plot(oh_centers[good], median_dtm[good], 'k-', lw=3, label='Median')
        ax.plot(oh_centers[good], median_dtm[good], 'w-', lw=1.5)
        
        # MW reference line (DtM ~ 0.5) and solar metallicity
        ax.axhline(np.log10(0.5), color='cyan', ls='--', lw=1.5, alpha=0.7)
        ax.axvline(8.69, color='cyan', ls=':', lw=1.5, alpha=0.7)  # Solar 12+log(O/H)
        
        # Labels and styling
        ax.set_xlim(6, 10)
        ax.set_ylim(-1, 0.5)
        ax.set_xlabel(r'$12 + \log({\rm O/H})$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_{\rm metals})$')
        ax.set_title(z_label, fontsize=14)
        
        # Add galaxy count
        ax.text(0.05, 0.95, f'N = {len(w):,}', transform=ax.transAxes,
               ha='left', va='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add colorbar
    cbar_ax6 = fig6.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar6 = fig6.colorbar(im6, cax=cbar_ax6)
    cbar6.set_label(r'$N_{\rm gal}$', color='white')
    cbar6.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar6.ax.axes, 'yticklabels'), color='white')
    
    fig6.suptitle('Dust-to-Metal Ratio vs Metallicity Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    outfile6 = os.path.join(OutputDir, 'DustToMetal_Metallicity_Evolution' + OutputFormat)
    fig6.savefig(outfile6, facecolor=fig6.get_facecolor())
    print(f'  Saved dust-to-metal vs metallicity figure to: {outfile6}\n')
    plt.close(fig6)

    # ==================================================================
    # NEW FIGURE: Dust and Star Formation History of MW-type Galaxy
    # ==================================================================
    print(f'  Creating MW-type galaxy dust history figure...')
    
    # Lookback times for each snapshot (Gyr)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=73, Om0=0.25)
    lookback_times = np.array([cosmo.lookback_time(z).value for z in redshifts])  # Gyr
    cosmic_times = cosmo.age(0).value - lookback_times  # Age of universe at each z
    
    # Helper function to read multiple arrays from all model files for a snapshot
    def read_snap_from_all_files(dirpath, snap_str, params):
        """Read multiple params from all model_*.hdf5 files, return dict of concatenated arrays."""
        result = {p: [] for p in params}
        i = 0
        while True:
            fname = os.path.join(dirpath, f'model_{i}.hdf5')
            if not os.path.exists(fname):
                break
            with h5.File(fname, 'r') as f:
                if snap_str in f:
                    for p in params:
                        if p in f[snap_str]:
                            result[p].append(np.array(f[snap_str][p]))
            i += 1
        # Concatenate arrays
        for p in params:
            if result[p]:
                result[p] = np.concatenate(result[p])
            else:
                result[p] = None
        return result
    
    # Check if DustDot arrays are available (check first file)
    has_dustdot = False
    with h5.File(os.path.join(DirName, 'model_0.hdf5'), 'r') as f:
        has_dustdot = 'Snap_63/DustDotForm' in f
    
    if has_dustdot:
        print(f'    Using DustDot arrays from HDF5 output')
    else:
        print(f'    DustDot arrays not found, estimating rates from physics')
    
    # Get z=0 data from ALL files to identify MW-like galaxies
    z0_params = ['StellarMass', 'GalaxyIndex', 'SfrDisk', 'SfrBulge']
    z0_data = read_snap_from_all_files(DirName, f'Snap_{LastSnap}', z0_params)
    
    sm_z0 = z0_data['StellarMass'] * 1e10 / Hubble_h
    gi_z0 = z0_data['GalaxyIndex']
    sfr_z0 = z0_data['SfrDisk'] + z0_data['SfrBulge']
    
    # MW-like: stellar mass between 3e10 and 1e11 Msun, AND star-forming
    mw_mask = (sm_z0 > 3e10) & (sm_z0 < 1e11) & (sfr_z0 > 0.1)
    mw_indices = gi_z0[mw_mask]
    
    print(f'    Found {len(mw_indices)} star-forming MW-like galaxies at z=0')
    
    # Limit to first 100 MW galaxies for computing median
    n_mw = min(100, len(mw_indices))
    mw_sample = mw_indices[:n_mw]
    
    # Storage arrays for all MW galaxies across time
    all_dust_mass = []
    all_sfr = []
    all_sfr_p16 = []  # 16th percentile of SFR
    all_sfr_p84 = []  # 84th percentile of SFR
    all_metallicity = []
    all_cold_gas = []
    all_metals_cold = []
    all_h2_gas = []
    all_dustdot_form = []     # From HDF5
    all_dustdot_growth = []   # From HDF5
    all_dustdot_destruct = [] # From HDF5
    snap_has_data = []
    
    # Parameters to read for each snapshot
    snap_params = ['GalaxyIndex', 'StellarMass', 'ColdDust', 'ColdGas', 
                   'MetalsColdGas', 'SfrDisk', 'SfrBulge', 'H2gas']
    if has_dustdot:
        snap_params.extend(['DustDotForm', 'DustDotGrowth', 'DustDotDestruct'])
    
    # Read data for each relevant snapshot from ALL files
    for snap in range(FirstSnap, LastSnap + 1):
        snap_str = f'Snap_{snap}'
        
        snap_data = read_snap_from_all_files(DirName, snap_str, snap_params)
        
        if snap_data['GalaxyIndex'] is None:
            all_dust_mass.append(np.nan)
            all_sfr.append(np.nan)
            all_sfr_p16.append(np.nan)
            all_sfr_p84.append(np.nan)
            all_metallicity.append(np.nan)
            all_cold_gas.append(np.nan)
            all_metals_cold.append(np.nan)
            all_h2_gas.append(np.nan)
            all_dustdot_form.append(np.nan)
            all_dustdot_growth.append(np.nan)
            all_dustdot_destruct.append(np.nan)
            snap_has_data.append(False)
            continue
        
        gi = snap_data['GalaxyIndex']
        sm = snap_data['StellarMass'] * 1e10 / Hubble_h
        cd = snap_data['ColdDust'] * 1e10 / Hubble_h
        cg = snap_data['ColdGas'] * 1e10 / Hubble_h
        mcg = snap_data['MetalsColdGas'] * 1e10 / Hubble_h
        sfr_disk = snap_data['SfrDisk']
        sfr_bulge = snap_data['SfrBulge']
        h2 = snap_data['H2gas'] * 1e10 / Hubble_h
        
        # Read DustDot arrays if available (sum over STEPS for total rate)
        if has_dustdot and snap_data['DustDotForm'] is not None:
            dustdot_form = snap_data['DustDotForm']
            dustdot_growth = snap_data['DustDotGrowth']
            dustdot_destruct = snap_data['DustDotDestruct']
            # Sum over steps (or use mean - they're rates per step)
            ddform = np.mean(dustdot_form, axis=1)  # Mean rate over steps
            ddgrowth = np.mean(dustdot_growth, axis=1)
            dddestruct = np.mean(dustdot_destruct, axis=1)
        else:
            ddform = None
            ddgrowth = None
            dddestruct = None
        
        # Find MW progenitors at this snapshot
        dust_this_snap = []
        sfr_this_snap = []
        Z_this_snap = []
        cg_this_snap = []
        mcg_this_snap = []
        h2_this_snap = []
        ddform_this_snap = []
        ddgrowth_this_snap = []
        dddestruct_this_snap = []
        
        for mw_gi in mw_sample:
            match = np.where(gi == mw_gi)[0]
            if len(match) > 0:
                idx = match[0]
                dust_this_snap.append(cd[idx])
                sfr_this_snap.append(sfr_disk[idx] + sfr_bulge[idx])
                if cg[idx] > 0:
                    Z_this_snap.append(mcg[idx] / cg[idx])
                else:
                    Z_this_snap.append(0.0)
                cg_this_snap.append(cg[idx])
                mcg_this_snap.append(mcg[idx])
                h2_this_snap.append(h2[idx])
                if has_dustdot and ddform is not None:
                    ddform_this_snap.append(ddform[idx])
                    ddgrowth_this_snap.append(ddgrowth[idx])
                    dddestruct_this_snap.append(dddestruct[idx])
        
        if len(dust_this_snap) > 0:
            all_dust_mass.append(np.median(dust_this_snap))
            all_sfr.append(np.median(sfr_this_snap))
            all_sfr_p16.append(np.percentile(sfr_this_snap, 16))
            all_sfr_p84.append(np.percentile(sfr_this_snap, 84))
            all_metallicity.append(np.median(Z_this_snap))
            all_cold_gas.append(np.median(cg_this_snap))
            all_metals_cold.append(np.median(mcg_this_snap))
            all_h2_gas.append(np.median(h2_this_snap))
            if has_dustdot and len(ddform_this_snap) > 0:
                all_dustdot_form.append(np.median(ddform_this_snap))
                all_dustdot_growth.append(np.median(ddgrowth_this_snap))
                all_dustdot_destruct.append(np.median(dddestruct_this_snap))
            else:
                all_dustdot_form.append(np.nan)
                all_dustdot_growth.append(np.nan)
                all_dustdot_destruct.append(np.nan)
            snap_has_data.append(True)
        else:
            all_dust_mass.append(np.nan)
            all_sfr.append(np.nan)
            all_sfr_p16.append(np.nan)
            all_sfr_p84.append(np.nan)
            all_metallicity.append(np.nan)
            all_cold_gas.append(np.nan)
            all_metals_cold.append(np.nan)
            all_h2_gas.append(np.nan)
            all_dustdot_form.append(np.nan)
            all_dustdot_growth.append(np.nan)
            all_dustdot_destruct.append(np.nan)
            snap_has_data.append(False)
    
    # Convert to arrays
    dust_mass = np.array(all_dust_mass)
    sfr = np.array(all_sfr)
    sfr_p16 = np.array(all_sfr_p16)
    sfr_p84 = np.array(all_sfr_p84)
    metallicity = np.array(all_metallicity)
    cold_gas = np.array(all_cold_gas)
    metals_cold = np.array(all_metals_cold)
    h2_gas = np.array(all_h2_gas)
    snap_has_data = np.array(snap_has_data)
    dustdot_form = np.array(all_dustdot_form)
    dustdot_growth = np.array(all_dustdot_growth)
    dustdot_destruct = np.array(all_dustdot_destruct)
    
    # Check if we have valid DustDot data
    has_valid_dustdot = has_dustdot and np.any(~np.isnan(dustdot_form))
    
    if has_valid_dustdot:
        # Use actual dust rates from simulation
        print(f'    Using actual DustDot values from SAGE output')
        prod_rate = dustdot_form  # Dust formation rate from stellar ejecta (Msun/yr)
        accretion_rate = dustdot_growth  # Dust growth rate in ISM (Msun/yr)
        destruction_rate = dustdot_destruct  # Dust destruction rate (Msun/yr)
    else:
        # Fallback: Estimate rates from physics (original method)
        print(f'    Using estimated dust rates from physics approximations')
        
        # Calculate dust rates from mass differences
        # Use cosmic time differences (Gyr)
        dt = np.diff(cosmic_times)  # Gyr between snapshots
        
        # Total dust rate: dM_dust/dt
        total_dust_rate = np.zeros(len(dust_mass))
        total_dust_rate[1:] = np.diff(dust_mass) / (dt * 1e9)  # Msun/yr
        
        # Estimate component rates based on physics:
        # 1. Production rate ~ delta_eff × Yield × SFR
        #    delta_eff ~ 0.27 (weighted AGB+SNII+SNIa: 0.3×0.6 + 0.6×0.15 + 0.1×0.03)
        #    Yield ~ 0.03
        delta_eff = 0.27
        yield_frac = 0.03
        prod_rate = delta_eff * yield_frac * sfr  # Msun/yr
        Z_solar = 0.02
        
        # 2. Accretion rate ~ (1 - DtM)^2 × f_H2 × M_dust / tau_acc
        #    tau_acc = 50 Myr × (Z_solar/Z)  [from parameter file]
        #    Squared term matches model (Asano+13 eq 20 with self-regulation)
        tau_acc_0 = 50e6  # 50 Myr in years (DustAccretionTimescale from .par)
        f_h2 = np.where(cold_gas > 0, h2_gas / cold_gas, 0)
        f_h2 = np.clip(f_h2, 0, 1)
        # Default f_h2 = 0.5 when H2 tracking unavailable (matches model)
        f_h2 = np.where(f_h2 < 1e-10, 0.5, f_h2)
        dtm = np.where(metals_cold > 0, dust_mass / metals_cold, 0)
        dtm = np.clip(dtm, 0, 1)
        tau_acc = np.where(metallicity > 0, tau_acc_0 * Z_solar / metallicity, tau_acc_0)
        accretion_rate = (1 - dtm)**2 * f_h2 * dust_mass / tau_acc  # Msun/yr
        
        # 3. Destruction rate ~ eta × m_swept × R_SN × DtG
        #    R_SN ~ SFR / 100 Msun (1 SN per 100 Msun)
        #    Note: Using higher efficiency (0.5) to match dusty-sage which includes
        #    both SNII and SNIa destruction. The model code uses 0.1 for SNII only.
        eta_sn = 0.5  # effective destruction efficiency (SNII + SNIa contribution)
        Z_solar = 0.02
        m_swept = 1535 * (metallicity / Z_solar + 0.039)**(-0.289)  # Msun
        R_sn = sfr / 100.0  # SN rate (per year)
        dtg = np.where(cold_gas > 0, dust_mass / cold_gas, 0)
        # Destruction timescale
        tau_dest = np.where((R_sn > 0) & (cold_gas > 0), cold_gas / (eta_sn * m_swept * R_sn), 1e12)
        destruction_rate = dust_mass / tau_dest  # Msun/yr
    
    # Total dust formation rate = production + growth (sources only)
    total_formation_rate = prod_rate + accretion_rate
    
    # Create figure
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    
    # Select valid data points (after early snapshots, with positive dust)
    valid = (snap_has_data) & (lookback_times < 13) & (dust_mass > 0) & (prod_rate > 0)
    
    # Plot dust rates vs lookback time (dusty-sage style)
    ax7.plot(lookback_times[valid], total_formation_rate[valid], 'w-', lw=3, 
             label='Total dust formation')
    ax7.plot(lookback_times[valid], prod_rate[valid], 'g--', lw=2.5, 
             label='Formation in stellar ejecta')
    ax7.plot(lookback_times[valid], accretion_rate[valid], 'b-.', lw=2.5, 
             label='Growth in the ISM')
    ax7.plot(lookback_times[valid], destruction_rate[valid], 'r:', lw=2.5, 
             label='Destruction by SN')
    
    # Secondary y-axis for SFR
    ax7_right = ax7.twinx()
    ax7_right.plot(lookback_times[valid], sfr[valid], 'm-', lw=10, alpha=0.3, label='Star formation rate')
    
    # Labels
    ax7.set_xlabel('Lookback time (Gyr)', fontsize=14)
    ax7.set_ylabel(r'$\log_{10}$ Dust rate ($M_\odot$ yr$^{-1}$)', fontsize=14)
    ax7_right.set_ylabel(r'SFR ($M_\odot$ yr$^{-1}$)', fontsize=14, color='magenta')
    ax7_right.tick_params(axis='y', labelcolor='magenta')
    
    # Add redshift axis on top
    ax7_top = ax7.twiny()
    z_ticks = [0, 0.2, 0.5, 1, 2, 4]
    lb_ticks = [cosmo.lookback_time(z).value for z in z_ticks]
    ax7_top.set_xlim(ax7.get_xlim())
    ax7_top.set_xticks(lb_ticks)
    ax7_top.set_xticklabels([f'{z}' for z in z_ticks])
    ax7_top.set_xlabel('redshift', fontsize=12)
    
    # Limits - lookback time from 0 (today) to ~13 Gyr (early universe)
    ax7.set_xlim(0, 13)
    ax7.set_yscale('log')
    ax7.set_ylim(1e-4, 1)
    ax7_right.set_yscale('log')
    ax7_right.set_ylim(0.3, 30)  # Better range for MW SFR (~1-10 Msun/yr)
    
    # Legend
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_right.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=11)
    
    ax7.set_title('Dust and Star Formation History of MW-type Galaxy', fontsize=16)
    
    plt.tight_layout()
    
    outfile7 = os.path.join(OutputDir, 'MW_Dust_History' + OutputFormat)
    fig7.savefig(outfile7, facecolor=fig7.get_facecolor())
    print(f'  Saved MW dust history figure to: {outfile7}\n')
    plt.close(fig7)


if __name__ == '__main__':
    main()

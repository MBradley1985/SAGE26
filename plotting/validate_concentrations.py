#!/usr/bin/env python
"""
Validate halo concentrations and g_max from SAGE26 BK25 FFB output.

Compares the code's lookup-table concentrations against Colossus,
checks g_max computation, and shows FFB activation statistics.

Usage:
    python plotting/validate_concentrations.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py as h5

# Optional: try to load a nice style
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_FILE = os.path.join(SCRIPT_DIR, 'ciaran_ohare_palatino_sty.mplstyle')
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'output', 'millennium_ffb_bk25')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'plots')
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
    print("=" * 70)
    print("SAGE26 Concentration & g_max Validation")
    print("=" * 70)

    header = read_header(MODEL_DIR)
    h = header['hubble_h']
    Om = header['omega_matter']
    OL = header['omega_lambda']
    redshifts = header['redshifts']
    output_snaps = header['output_snaps']
    print(f"Cosmology: h={h}, Omega_m={Om}, Omega_L={OL}")
    print(f"N output snapshots: {len(output_snaps)}")
    print()

    props = ['Mvir', 'Rvir', 'Concentration', 'g_max', 'FFBRegime', 'Type',
             'StellarMass', 'ColdGas']

    # Select snapshots spanning a range of redshifts
    snap_targets = [(63, 0.0), (50, 0.4), (40, 1.1), (30, 2.4),
                    (25, 3.5), (20, 5.3), (15, 8.2), (12, 10.1), (10, 11.9)]
    # Filter to ones that exist
    snap_list = []
    for sn, z_approx in snap_targets:
        if sn in output_snaps:
            snap_list.append(sn)

    # ======================================================================
    # Collect data across snapshots
    # ======================================================================
    all_data = {}
    for snap in snap_list:
        data = read_snap(MODEL_DIR, snap, props)
        if 'Mvir' not in data or len(data['Mvir']) == 0:
            continue
        z = redshifts[snap]
        # Physical quantities
        Mvir_Msun = data['Mvir'] * 1e10 / h
        logM = np.log10(np.clip(Mvir_Msun * h, 1e6, None))  # log10(Msun/h) for table lookup
        Rvir_Mpc = data['Rvir'] / h  # physical Mpc
        Rvir_pc = Rvir_Mpc * 1e6     # physical pc
        c = data['Concentration']
        g_max_code = data['g_max']
        ffb = data['FFBRegime']
        gtype = data['Type']

        # Only centrals (Type 0) with valid data
        mask = (gtype == 0) & (data['Mvir'] > 0) & (c > 0)

        all_data[snap] = {
            'z': z,
            'logM': logM[mask],
            'Mvir_Msun': Mvir_Msun[mask],
            'Rvir_pc': Rvir_pc[mask],
            'c': c[mask],
            'g_max_code': g_max_code[mask],
            'ffb': ffb[mask],
            'n_total': len(data['Mvir']),
            'n_centrals': int(mask.sum()),
        }

    # ======================================================================
    # Print summary statistics
    # ======================================================================
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
    # Figure 1: Concentration vs Mass at multiple redshifts
    # ======================================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, snap in enumerate(sorted(all_data.keys(), reverse=True)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        d = all_data[snap]
        z = d['z']

        # Scatter plot: SAGE concentrations
        normal = d['ffb'] == 0
        ffb_mask = d['ffb'] == 1
        ax.scatter(d['logM'][normal], d['c'][normal], s=1, alpha=0.15,
                   color='C0', rasterized=True, label='Normal')
        if ffb_mask.any():
            ax.scatter(d['logM'][ffb_mask], d['c'][ffb_mask], s=4, alpha=0.5,
                       color='C3', rasterized=True, label='FFB')

        # Colossus reference line
        logM_grid = np.linspace(max(d['logM'].min(), 8.0), min(d['logM'].max(), 16.0), 100)
        c_ref = get_colossus_concentrations(logM_grid, z, h, Om, OL)
        if c_ref is not None:
            ax.plot(logM_grid, c_ref, 'k-', lw=2, label='Ishiyama+21')

        # Median binned
        logM_bins = np.arange(8, 16.1, 0.3)
        bin_centers = 0.5 * (logM_bins[:-1] + logM_bins[1:])
        bin_idx = np.digitize(d['logM'], logM_bins)
        c_median = np.array([np.median(d['c'][bin_idx == i])
                             if (bin_idx == i).sum() > 5 else np.nan
                             for i in range(1, len(logM_bins))])
        valid = ~np.isnan(c_median)
        ax.plot(bin_centers[valid], c_median[valid], 'o-', color='C1',
                ms=4, lw=1.5, label='SAGE median')

        ax.set_title(f'z = {z:.1f}  (snap {snap})', fontsize=12)
        ax.set_ylim(0.5, 30)
        ax.set_yscale('log')
        ax.set_xlim(8, 15)
        ax.grid(False)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

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

        # Convert code g_max to g_max/G in Msun/pc^2
        # g_code = G_code * M_code / R_code^2  (in code accel units)
        # g_phys/G = M_phys / R_phys^2 = (M_code*1e10/h) / (R_code/h * 1e6 pc)^2
        #          = M_code * 1e10 * h / (R_code^2 * 1e12)
        # So: g/G [Msun/pc^2] = g_code * (1e10/h) / (1/h * 1e6)^2 ... tricky
        # Easier: just compare code g_max values directly
        # g_max_recomputed_code = G_code * Mvir_code / Rvir_code^2 * c^2/(2*mu(c))
        # But we don't have G_code here easily. Instead compare the physical quantity.

        # Scatter: Python g_max/G vs SAGE g_max/G
        # We need to convert code g_max to physical units.
        # Actually, let's compare the ratio g_max / g_crit directly
        # g_crit/G = 3100 Msun/pc^2
        ratio_py = g_max_over_G_py / G_CRIT_OVER_G

        # For code: we know g_crit_code, so g_max_code/g_crit_code = g_max_phys/g_crit_phys
        # since both are in the same code units.
        # But we can't easily get g_crit_code from here without recomputing G_code.
        # Instead, FFBRegime tells us if g_max > g_crit in code.
        # Let's just compare the physical recomputation.

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
    ax1.plot(z_arr, f_ffb_arr, 'o-', color='C3', lw=2, ms=6, label='SAGE26 BK25')

    # Li+24 FFB fraction from actual SAGE25 run output
    LI24_MODEL = os.path.join(PROJECT_DIR, 'output', 'millennium')
    li24_z_arr = []
    li24_f_ffb = []
    for snap in sorted(all_data.keys()):
        z = redshifts[snap]
        li_data = read_snap(LI24_MODEL, snap, ['Mvir', 'FFBRegime', 'Type'])
        if 'Mvir' not in li_data or len(li_data['Mvir']) == 0:
            continue
        li_cen = (li_data['Type'] == 0) & (li_data['Mvir'] > 0)
        n_cen = li_cen.sum()
        if n_cen == 0:
            continue
        n_ffb = (li_data['FFBRegime'][li_cen] == 1).sum()
        li24_z_arr.append(z)
        li24_f_ffb.append(n_ffb / n_cen)
    ax1.plot(li24_z_arr, li24_f_ffb, 's--', color='C0', lw=2, ms=5, label='SAGE25 Li+24')

    ax1.set_xlabel('Redshift $z$', fontsize=13)
    ax1.set_ylabel('FFB Fraction (centrals)', fontsize=13)
    ax1.set_title('FFB Activation vs Redshift', fontsize=13)
    ax1.set_xlim(-0.5, max(z_arr) + 0.5)
    ymax = max(max(f_ffb_arr), max(li24_f_ffb)) if li24_f_ffb else max(f_ffb_arr)
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
    # by root-finding at each redshift: find M_200c where g_max(M, c(M,z), z) = g_crit
    z_theory = np.linspace(4, 15, 50)
    logM_theory = np.full_like(z_theory, np.nan)
    try:
        from colossus.cosmology import cosmology as colossus_cosmo
        from colossus.halo import concentration as colossus_conc
        from scipy.optimize import brentq
        colossus_cosmo.setCosmology('mill_thresh', flat=True, H0=h*100,
                                     Om0=Om, Ob0=0.045, sigma8=0.9, ns=1.0,
                                     relspecies=False)
        Hubble_code = 100.0  # H0 in code units (see macros.h)
        G_code = 43.0071
        Msun_code = 1.989e33 / 1.989e43
        pc_code = 3.08568e18 / 3.08568e24
        g_crit_code = G_code * 3100.0 * Msun_code / pc_code**2 / h

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
    ax3.plot(z_arr, c_median_arr, 'o-', color='C2', lw=2, ms=6)
    ax3.set_xlabel('Redshift $z$', fontsize=13)
    ax3.set_ylabel('Median Concentration (centrals)', fontsize=13)
    ax3.set_title('Median Concentration vs Redshift', fontsize=13)
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
    # Verify that g_max from the code (in code units) is proportional to
    # the physical g_max/G we recompute from Mvir, Rvir, c.
    # They should be perfectly proportional since g_code = G_code * M_code/R_code^2 * f(c)
    # and g_phys/G = M_phys/R_phys^2 * f(c), with a constant ratio between them.

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, snap in enumerate(check_snaps):
        ax = axes[idx]
        d = all_data[snap]
        z = d['z']

        # Physical g_max/G
        g_vir_over_G = d['Mvir_Msun'] / d['Rvir_pc']**2
        g_max_phys = g_vir_over_G * d['c']**2 / (2.0 * mu_nfw(d['c']))

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
    # Figure 6: Mvir vs Redshift coloured by concentration (side-by-side)
    # ======================================================================
    # Collect ALL galaxies (not just centrals) for this figure
    fig6_z = []
    fig6_logM = []
    fig6_c = []
    fig6_ffb = []
    for snap in output_snaps:
        z = redshifts[snap]
        data = read_snap(MODEL_DIR, snap, props)
        if 'Mvir' not in data or len(data['Mvir']) == 0:
            continue
        c_arr = data['Concentration']
        mask = (data['Mvir'] > 0) & (c_arr > 0)
        logM_all = np.log10(np.clip(data['Mvir'][mask] * 1e10, 1e6, None))  # Msun/h
        fig6_z.append(np.full(mask.sum(), z))
        fig6_logM.append(logM_all)
        fig6_c.append(c_arr[mask])
        fig6_ffb.append(data['FFBRegime'][mask])
    fig6_z = np.concatenate(fig6_z)
    fig6_logM = np.concatenate(fig6_logM)
    fig6_c = np.concatenate(fig6_c)
    fig6_ffb = np.concatenate(fig6_ffb)

    c_vmin, c_vmax = np.percentile(fig6_c, [2, 98])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Left panel: all galaxies coloured by concentration ---
    sc1 = ax1.scatter(fig6_z, fig6_logM, c=fig6_c, s=2, alpha=0.3,
                      cmap='viridis', vmin=c_vmin, vmax=c_vmax, rasterized=True)
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.02, alpha=1.0)
    cb1.solids.set_alpha(1.0)
    cb1.set_label('Concentration $c$', fontsize=12)
    ax1.set_xlabel('Redshift $z$', fontsize=13)
    ax1.set_ylabel(r'$\log_{10}(M_{200c}\ /\ M_\odot\,h^{-1})$', fontsize=13)
    ax1.set_title('All Galaxies', fontsize=13)
    ax1.set_ylim(10, 15)

    # --- Right panel: normal = grey, FFB = coloured by concentration ---
    normal = fig6_ffb == 0
    ffb = fig6_ffb == 1

    ax2.scatter(fig6_z[normal], fig6_logM[normal], c='0.8', s=2, alpha=0.2,
                rasterized=True, label='Normal')
    sc2 = ax2.scatter(fig6_z[ffb], fig6_logM[ffb], c=fig6_c[ffb], s=60, alpha=0.9,
                      marker='*', cmap='viridis', vmin=c_vmin, vmax=c_vmax,
                      edgecolors='k', linewidths=0.3, rasterized=True, zorder=5,
                      label='FFB')
    cb2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label('Concentration $c$', fontsize=12)
    # Overplot g_max = g_crit threshold lines for fixed concentrations
    # At fixed c, g_max = g_crit gives:
    #   M_thresh = [g_crit / (G * (200*4pi/3 * rho_crit)^(2/3) * c^2/(2*mu(c)))]^3
    Hubble_code = 100.0
    G_code = 43.0071
    Msun_code = 1.989e33 / 1.989e43
    pc_code = 3.08568e18 / 3.08568e24
    g_crit_code = G_code * 3100.0 * Msun_code / pc_code**2 / h

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
            # g_max = G * M / R^2 * c^2/(2*mu) = G * M^(1/3) * fac200^(2/3) * c^2/(2*mu)
            # solve for M: M = (g_crit / (G * fac200^(2/3) * c^2/(2*mu)))^3
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
    ax2.plot(z_th, logM_li24, 'b--', lw=2, label=r'Li+24 / SAGE25')

    # 2) BK25 (Planck, virial): log10(M/Msun) = 10.8 - 6*log10((1+z)/11), convert to Msun/h
    logM_bk25 = 10.8 + np.log10(h) - 6.0 * np.log10((1 + z_th) / 11.0)
    ax2.plot(z_th, logM_bk25, 'r--', lw=2, label=r'BK25 (Planck, vir)')

    # 3) SAGE26 BK25 (Millennium, 200c): root-find g_max = g_crit at each z
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

        logM_sage26 = np.full_like(z_th, np.nan)
        for i, zv in enumerate(z_th):
            try:
                logM_sage26[i] = brentq(g_max_minus_g_crit_fig6, 8.0, 15.0, args=(zv,))
            except ValueError:
                pass
        valid_s26 = ~np.isnan(logM_sage26)
        ax2.plot(z_th[valid_s26], logM_sage26[valid_s26], 'r-', lw=2.5,
                 label=r'SAGE26 BK25 (Mill, 200c)')
    except ImportError:
        pass

    ax2.set_xlabel('Redshift $z$', fontsize=13)
    ax2.set_title('FFB Galaxies Highlighted', fontsize=13)
    ax2.legend(fontsize=8, loc='upper right', ncol=3)
    ax2.grid(False)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'mvir_vs_redshift_concentration.pdf')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

    print()
    print("=" * 70)
    print("Validation complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()

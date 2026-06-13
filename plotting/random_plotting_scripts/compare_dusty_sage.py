#!/usr/bin/env python
"""
Compare dusty-sage binary output with SAGE26 HDF5 output.
Focuses on dust properties: ColdDust, HotDust, EjectedDust.

Usage:
    python plotting/presentation_plots/compare_dusty_sage.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from os.path import getsize as getFileSize

# ==================== Configuration ====================

# Paths
DUSTY_SAGE_DIR = '/Users/mbradley/Documents/PhD/dusty-sage/src/auxdata/trees/mini-millennium/'
SAGE26_DIR = './output/millennium/'

# Simulation constants
HUBBLE_H = 0.73
BOX_SIZE = 62.5  # h^-1 Mpc
VOLUME = (BOX_SIZE / HUBBLE_H)**3  # Mpc^3
MASS_CONVERT = 1.0e10 / HUBBLE_H  # Convert to Msun

# Redshift snapshots to compare
SNAP_Z0 = 63
REDSHIFT_MAP = {
    63: 0.000, 37: 1.386, 32: 2.070, 27: 3.060,
    23: 4.179, 20: 5.289, 18: 6.197, 16: 7.272
}

# ==================== dusty-sage Binary Reader ====================

def get_dusty_sage_dtype():
    """Return the numpy dtype for dusty-sage GALAXY_OUTPUT structure."""
    return np.dtype([
        ('SnapNum', np.int32),
        ('Type', np.int32),
        ('GalaxyIndex', np.int64),
        ('CentralGalaxyIndex', np.int64),
        ('SAGEHaloIndex', np.int32),
        ('SAGETreeIndex', np.int32),
        ('SimulationHaloIndex', np.int64),
        ('mergeType', np.int32),
        ('mergeIntoID', np.int32),
        ('mergeIntoSnapNum', np.int32),
        ('dT', np.float32),
        ('Pos', (np.float32, 3)),
        ('Vel', (np.float32, 3)),
        ('Spin', (np.float32, 3)),
        ('Len', np.int32),
        ('Mvir', np.float32),
        ('CentralMvir', np.float32),
        ('Rvir', np.float32),
        ('Vvir', np.float32),
        ('Vmax', np.float32),
        ('VelDisp', np.float32),
        ('ColdGas', np.float32),
        ('f_H2', np.float32),
        ('f_HI', np.float32),
        ('cf', np.float32),
        ('Zp', np.float32),
        ('Pressure', np.float32),
        ('StellarMass', np.float32),
        ('BulgeMass', np.float32),
        ('BulgeInstability', np.float32),
        ('HotGas', np.float32),
        ('EjectedMass', np.float32),
        ('BlackHoleMass', np.float32),
        ('ICS', np.float32),
        ('MetalsColdGas', np.float32),
        ('MetalsStellarMass', np.float32),
        ('MetalsBulgeMass', np.float32),
        ('MetalsHotGas', np.float32),
        ('MetalsEjectedMass', np.float32),
        ('MetalsICS', np.float32),
        ('ColdDust', np.float32),
        ('HotDust', np.float32),
        ('EjectedDust', np.float32),
        ('SfrDisk', np.float32),
        ('SfrBulge', np.float32),
        ('SfrDiskZ', np.float32),
        ('SfrBulgeZ', np.float32),
        ('SfrDiskDTG', np.float32),
        ('SfrBulgeDTG', np.float32),
        ('dustdotform', np.float32),
        ('dustdotgrowth', np.float32),
        ('dustdotdestruct', np.float32),
        ('DiskScaleRadius', np.float32),
        ('Cooling', np.float32),
        ('Heating', np.float32),
        ('QuasarModeBHaccretionMass', np.float32),
        ('TimeOfLastMajorMerger', np.float32),
        ('TimeOfLastMinorMerger', np.float32),
        ('OutflowRate', np.float32),
        ('infallMvir', np.float32),
        ('infallVvir', np.float32),
        ('infallVmax', np.float32),
    ], align=True)


def read_dusty_sage(model_dir, redshift_str, first_file=0, last_file=7):
    """Read dusty-sage binary output files for a given redshift."""
    
    Galdesc = get_dusty_sage_dtype()
    
    # Count total galaxies first
    TotNGals = 0
    valid_files = []
    
    for fnr in range(first_file, last_file + 1):
        fname = os.path.join(model_dir, f'model_z{redshift_str}_{fnr}')
        if not os.path.isfile(fname) or getFileSize(fname) == 0:
            continue
        
        with open(fname, 'rb') as fin:
            Ntrees = np.fromfile(fin, np.int32, 1)[0]
            NtotGals = np.fromfile(fin, np.int32, 1)[0]
            TotNGals += NtotGals
            valid_files.append((fname, Ntrees, NtotGals))
    
    if TotNGals == 0:
        print(f"  No galaxies found for z={redshift_str}")
        return None
    
    # Allocate and read
    G = np.empty(TotNGals, dtype=Galdesc)
    offset = 0
    
    for fname, Ntrees, NtotGals in valid_files:
        with open(fname, 'rb') as fin:
            np.fromfile(fin, np.int32, 1)  # Ntrees
            np.fromfile(fin, np.int32, 1)  # NtotGals
            np.fromfile(fin, np.dtype((np.int32, Ntrees)), 1)  # GalsPerTree
            G[offset:offset + NtotGals] = np.fromfile(fin, Galdesc, NtotGals)
            offset += NtotGals
    
    print(f"  dusty-sage z={redshift_str}: {TotNGals:,} galaxies")
    return G


def read_sage26(model_dir, snap):
    """Read SAGE26 HDF5 output for a given snapshot."""
    import glob
    
    # SAGE26 format: model.hdf5 or model_0.hdf5 with Snap_XX groups
    files = glob.glob(os.path.join(model_dir, 'model*.hdf5'))
    
    if not files:
        print(f"  SAGE26 snap={snap}: No HDF5 files found")
        return None
    
    snap_key = f'Snap_{snap}'
    all_data = {}
    total_ngals = 0
    
    for fpath in sorted(files):
        with h5py.File(fpath, 'r') as f:
            if snap_key not in f:
                continue
            gals = f[snap_key]
            
            # Check for any mass field to get count
            if 'StellarMass' not in gals:
                continue
            n = len(gals['StellarMass'])
            
            for key in ['StellarMass', 'ColdGas', 'HotGas', 'EjectedMass',
                        'MetalsColdGas', 'MetalsHotGas', 'MetalsEjectedMass',
                        'ColdDust', 'HotDust', 'EjectedDust', 'CGMDust',
                        'SfrDisk', 'SfrBulge']:
                if key in gals:
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(gals[key][:])
            total_ngals += n
    
    if total_ngals == 0:
        print(f"  SAGE26 snap={snap}: 0 galaxies")
        return None
    
    # Concatenate
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key])
    
    print(f"  SAGE26 snap={snap}: {total_ngals:,} galaxies")
    return all_data


# ==================== Comparison Plots ====================

def compute_dust_mass_function(dust_mass, volume, n_bins=30):
    """Compute dust mass function (phi = dN/dlogM/dV)."""
    
    # Filter valid masses
    valid = (dust_mass > 0) & np.isfinite(dust_mass)
    log_mass = np.log10(dust_mass[valid] * MASS_CONVERT)
    
    if len(log_mass) == 0:
        return None, None, None
    
    # Bin in log space
    bins = np.linspace(3, 10, n_bins + 1)
    hist, edges = np.histogram(log_mass, bins=bins)
    
    # Convert to phi (Mpc^-3 dex^-1)
    dlogM = edges[1] - edges[0]
    phi = hist / (volume * dlogM)
    
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Poisson errors
    phi_err = np.sqrt(hist) / (volume * dlogM)
    
    return centers, phi, phi_err


def plot_comparison(dusty_sage_data, sage26_data, redshift_str):
    """Create comparison plots for a single redshift."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'dusty-sage vs SAGE26 Comparison (z={redshift_str})', fontsize=14)
    
    # === Panel 1: Dust Mass Function ===
    ax = axes[0, 0]
    
    if dusty_sage_data is not None:
        total_dust_ds = dusty_sage_data['ColdDust'] + dusty_sage_data['HotDust'] + dusty_sage_data['EjectedDust']
        x, y, yerr = compute_dust_mass_function(total_dust_ds, VOLUME)
        if x is not None:
            ax.plot(x, np.log10(y + 1e-10), 'b-', lw=2, label='dusty-sage')
    
    if sage26_data is not None:
        total_dust_s26 = sage26_data['ColdDust'] + sage26_data.get('HotDust', 0) + sage26_data.get('EjectedDust', 0)
        x, y, yerr = compute_dust_mass_function(total_dust_s26, VOLUME)
        if x is not None:
            ax.plot(x, np.log10(y + 1e-10), 'r--', lw=2, label='SAGE26')
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm dust}/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\phi/{\rm Mpc}^{-3}\,{\rm dex}^{-1})$')
    ax.set_title('Dust Mass Function')
    ax.legend()
    ax.set_xlim(4, 9)
    ax.set_ylim(-6, 0)
    
    # === Panel 2: ColdDust vs StellarMass ===
    ax = axes[0, 1]
    
    if dusty_sage_data is not None:
        valid = (dusty_sage_data['StellarMass'] > 0) & (dusty_sage_data['ColdDust'] > 0)
        if np.sum(valid) > 0:
            x = np.log10(dusty_sage_data['StellarMass'][valid] * MASS_CONVERT)
            y = np.log10(dusty_sage_data['ColdDust'][valid] * MASS_CONVERT)
            ax.scatter(x[::10], y[::10], alpha=0.1, s=1, c='blue', label='dusty-sage')
    
    if sage26_data is not None:
        valid = (sage26_data['StellarMass'] > 0) & (sage26_data['ColdDust'] > 0)
        if np.sum(valid) > 0:
            x = np.log10(sage26_data['StellarMass'][valid] * MASS_CONVERT)
            y = np.log10(sage26_data['ColdDust'][valid] * MASS_CONVERT)
            ax.scatter(x[::10], y[::10], alpha=0.1, s=1, c='red', label='SAGE26')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm cold\,dust}/M_\odot)$')
    ax.set_title('Cold Dust vs Stellar Mass')
    ax.set_xlim(7, 12)
    ax.set_ylim(3, 9)
    
    # === Panel 3: Dust-to-Gas Ratio ===
    ax = axes[0, 2]
    
    if dusty_sage_data is not None:
        valid = (dusty_sage_data['ColdGas'] > 1e-6) & (dusty_sage_data['MetalsColdGas'] > 0)
        if np.sum(valid) > 0:
            Z = dusty_sage_data['MetalsColdGas'][valid] / dusty_sage_data['ColdGas'][valid]
            DTG = dusty_sage_data['ColdDust'][valid] / dusty_sage_data['ColdGas'][valid]
            mask = (DTG > 0) & (Z > 0)
            ax.scatter(np.log10(Z[mask]/0.02)[::10], np.log10(DTG[mask])[::10], 
                      alpha=0.1, s=1, c='blue', label='dusty-sage')
    
    if sage26_data is not None:
        valid = (sage26_data['ColdGas'] > 1e-6) & (sage26_data['MetalsColdGas'] > 0)
        if np.sum(valid) > 0:
            Z = sage26_data['MetalsColdGas'][valid] / sage26_data['ColdGas'][valid]
            DTG = sage26_data['ColdDust'][valid] / sage26_data['ColdGas'][valid]
            mask = (DTG > 0) & (Z > 0)
            ax.scatter(np.log10(Z[mask]/0.02)[::10], np.log10(DTG[mask])[::10],
                      alpha=0.1, s=1, c='red', label='SAGE26')
    
    ax.set_xlabel(r'$\log_{10}(Z/Z_\odot)$')
    ax.set_ylabel(r'$\log_{10}(D/G)$')
    ax.set_title('Dust-to-Gas vs Metallicity')
    ax.set_xlim(-2, 1)
    ax.set_ylim(-5, -1)
    
    # === Panel 4: Stellar Mass Function ===
    ax = axes[1, 0]
    
    def smf(mstar, volume):
        valid = mstar > 0
        logm = np.log10(mstar[valid] * MASS_CONVERT)
        bins = np.linspace(7, 12, 26)
        hist, edges = np.histogram(logm, bins=bins)
        dlogM = edges[1] - edges[0]
        phi = hist / (volume * dlogM)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, phi
    
    if dusty_sage_data is not None:
        x, y = smf(dusty_sage_data['StellarMass'], VOLUME)
        ax.plot(x, np.log10(y + 1e-10), 'b-', lw=2, label='dusty-sage')
    
    if sage26_data is not None:
        x, y = smf(sage26_data['StellarMass'], VOLUME)
        ax.plot(x, np.log10(y + 1e-10), 'r--', lw=2, label='SAGE26')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\phi/{\rm Mpc}^{-3}\,{\rm dex}^{-1})$')
    ax.set_title('Stellar Mass Function')
    ax.legend()
    ax.set_xlim(7, 12)
    ax.set_ylim(-6, 0)
    
    # === Panel 5: Dust Reservoirs ===
    ax = axes[1, 1]
    
    def reservoir_fracs(data, key_cold, key_hot, key_ej, is_structured=False):
        """Calculate median dust reservoir fractions."""
        if is_structured:
            # Numpy structured array (dusty-sage)
            cold = np.array(data[key_cold])
            hot = np.array(data[key_hot]) if key_hot in data.dtype.names else np.zeros_like(cold)
            ej = np.array(data[key_ej]) if key_ej in data.dtype.names else np.zeros_like(cold)
        else:
            # Dict (SAGE26)
            cold = data.get(key_cold, np.array([0]))
            hot = data.get(key_hot, np.array([0]))
            ej = data.get(key_ej, np.array([0]))
        
        if not isinstance(cold, np.ndarray) or len(cold) == 0:
            return 0, 0, 0
            
        total = cold + hot + ej
        valid = total > 1e-10
        if np.sum(valid) == 0:
            return 0, 0, 0
        
        f_cold = np.median(cold[valid] / total[valid])
        f_hot = np.median(hot[valid] / total[valid])
        f_ej = np.median(ej[valid] / total[valid])
        return f_cold, f_hot, f_ej
    
    labels = ['Cold', 'Hot', 'Ejected']
    x_pos = np.array([0, 1, 2])
    width = 0.35
    
    if dusty_sage_data is not None:
        fracs_ds = reservoir_fracs(dusty_sage_data, 'ColdDust', 'HotDust', 'EjectedDust', is_structured=True)
        ax.bar(x_pos - width/2, fracs_ds, width, label='dusty-sage', color='blue', alpha=0.7)
    
    if sage26_data is not None:
        fracs_s26 = reservoir_fracs(sage26_data, 'ColdDust', 'HotDust', 'EjectedDust', is_structured=False)
        ax.bar(x_pos + width/2, fracs_s26, width, label='SAGE26', color='red', alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median Fraction of Total Dust')
    ax.set_title('Dust Reservoir Distribution')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # === Panel 6: Star Formation Rate ===
    ax = axes[1, 2]
    
    if dusty_sage_data is not None:
        sfr = dusty_sage_data['SfrDisk'] + dusty_sage_data['SfrBulge']
        valid = (dusty_sage_data['StellarMass'] > 0) & (sfr > 0)
        if np.sum(valid) > 0:
            x = np.log10(dusty_sage_data['StellarMass'][valid] * MASS_CONVERT)
            y = np.log10(sfr[valid])
            ax.scatter(x[::10], y[::10], alpha=0.1, s=1, c='blue', label='dusty-sage')
    
    if sage26_data is not None:
        sfr = sage26_data['SfrDisk'] + sage26_data['SfrBulge']
        valid = (sage26_data['StellarMass'] > 0) & (sfr > 0)
        if np.sum(valid) > 0:
            x = np.log10(sage26_data['StellarMass'][valid] * MASS_CONVERT)
            y = np.log10(sfr[valid])
            ax.scatter(x[::10], y[::10], alpha=0.1, s=1, c='red', label='SAGE26')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})$')
    ax.set_title('Star Formation Main Sequence')
    ax.set_xlim(7, 12)
    ax.set_ylim(-4, 3)
    
    plt.tight_layout()
    return fig


# ==================== Main ====================

def main():
    print("=" * 60)
    print("dusty-sage vs SAGE26 Comparison")
    print("=" * 60)
    
    # Output directory
    os.makedirs('plotting/presentation_plots/output', exist_ok=True)
    
    # Redshifts to compare
    redshifts = ['0.000', '1.386', '2.070', '3.060']
    snaps = [63, 37, 32, 27]
    
    for z_str, snap in zip(redshifts, snaps):
        print(f"\nz = {z_str}:")
        
        # Read dusty-sage
        dusty_sage = read_dusty_sage(DUSTY_SAGE_DIR, z_str)
        
        # Read SAGE26
        sage26 = read_sage26(SAGE26_DIR, snap)
        
        if dusty_sage is None and sage26 is None:
            print("  No data for either model, skipping.")
            continue
        
        # Create comparison plot
        fig = plot_comparison(dusty_sage, sage26, z_str)
        
        outfile = f'plotting/presentation_plots/output/compare_z{z_str.replace(".", "p")}.pdf'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {outfile}")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

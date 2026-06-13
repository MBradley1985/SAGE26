#!/usr/bin/env python
"""
Compare concentration-mass relations from Colossus at masses and redshifts
relevant for FFB (Feedback-Free Bursts) in the BK25 framework.

Goal: Find which c-M relation gives c ~ 4 at z=10 for M ~ 10^10.8 Msun,
matching the BK25 assumption for the FFB threshold.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import concentration

# Set up Millennium cosmology
# cosmology.setCosmology('planck18')  # Close enough; we can also define custom
# For exact Millennium params:
# Millennium-like parameters (WMAP1-ish)
my_params = {
    'flat': True,
    'H0': 73.0,
    'Om0': 0.25,
    'Ob0': 0.045,
    'sigma8': 0.9,
    'ns': 1.0
}

# Register and set the custom cosmology
cosmo = cosmology.setCosmology('custom_millennium', **my_params)

# Mass and redshift grids
log_masses = np.linspace(8, 13, 100)  # log10(Msun/h)
masses = 10**log_masses

redshifts = [0, 2, 4, 6, 8, 10, 12, 15]

# Models to compare - these are the ones available in Colossus
# Focus on models that work at high redshift
models_to_try = [
    'diemer15',
    'dutton14',
    'duffy08',
    'ludlow16',
    'zhao09',
    'child18',
    'ishiyama21',
    'diemer19',
]

# ============================================================
# Plot 1: c vs M at several redshifts, one panel per model
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

colors_z = plt.cm.viridis(np.linspace(0, 0.9, len(redshifts)))

successful_models = []

for idx, model in enumerate(models_to_try):
    if idx >= len(axes):
        break
    ax = axes[idx]
    worked = False
    for iz, z in enumerate(redshifts):
        try:
            c_vals = concentration.concentration(masses, 'vir', z, model=model)
            ax.plot(log_masses, c_vals, color=colors_z[iz], label=f'z={z}')
            worked = True
        except Exception as e:
            # Some models don't support all redshifts
            pass

    if worked:
        successful_models.append(model)

    ax.axhline(4.0, color='red', ls='--', lw=2, alpha=0.7, label='c=4 (BK25)')
    ax.axvline(10.8, color='grey', ls=':', lw=1.5, alpha=0.7, label=r'$10^{10.8} M_\odot$')
    ax.set_title(model, fontsize=14)
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot h^{-1})$')
    ax.set_ylabel('c')
    ax.set_ylim(0, 25)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Concentration-Mass Relations from Colossus\n(red dashed = c=4, BK25 threshold assumption)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('colossus_cm_all_models.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved colossus_cm_all_models.pdf")
print(f"Successful models: {successful_models}")

# ============================================================
# Plot 2: c at the BK25 threshold mass (10^10.8 Msun) vs redshift
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

M_thresh = 10**10.8  # Msun (approximate BK25 threshold mass)
z_fine = np.linspace(0, 15, 100)

colors_model = plt.cm.tab10(np.linspace(0, 1, len(models_to_try)))

for im, model in enumerate(models_to_try):
    c_vs_z = []
    z_valid = []
    for z in z_fine:
        try:
            c_val = concentration.concentration(M_thresh, 'vir', z, model=model)
            c_vs_z.append(c_val)
            z_valid.append(z)
        except:
            pass
    if len(z_valid) > 0:
        ax.plot(z_valid, c_vs_z, color=colors_model[im], lw=2.5, label=model)

ax.axhline(4.0, color='red', ls='--', lw=2, alpha=0.7, label='c=4 (BK25 assumption)')
ax.axhline(7.0, color='orange', ls=':', lw=2, alpha=0.7, label='c=7 (current cap)')
ax.axvline(10, color='grey', ls=':', alpha=0.5)
ax.set_xlabel('Redshift z', fontsize=14)
ax.set_ylabel('Concentration c', fontsize=14)
ax.set_title(r'Concentration at $M_{\rm vir} = 10^{10.8}\,M_\odot$ vs Redshift', fontsize=15)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 20)
ax.set_xlim(0, 15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('colossus_cm_at_threshold.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Saved colossus_cm_at_threshold.pdf")

# ============================================================
# Plot 3: c vs M at z=10 specifically — the key FFB redshift
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

for im, model in enumerate(models_to_try):
    try:
        c_vals = concentration.concentration(masses, 'vir', 10.0, model=model)
        ax.plot(log_masses, c_vals, color=colors_model[im], lw=2.5, label=model)
    except Exception as e:
        print(f"  {model} failed at z=10: {e}")

ax.axhline(4.0, color='red', ls='--', lw=2, alpha=0.7, label='c=4 (BK25)')
ax.axvline(10.8, color='grey', ls=':', lw=1.5, alpha=0.7, label=r'$M = 10^{10.8} M_\odot$')
ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot h^{-1})$', fontsize=14)
ax.set_ylabel('Concentration c', fontsize=14)
ax.set_title('Concentration-Mass at z=10', fontsize=15)
ax.legend(fontsize=11)
ax.set_ylim(0, 15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('colossus_cm_z10.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Saved colossus_cm_z10.pdf")

# ============================================================
# Print table of c values at key mass/redshift combinations
# ============================================================
print("\n" + "="*80)
print("Concentration values at key (M, z) for FFB")
print("="*80)

key_masses_log = [9.0, 10.0, 10.8, 11.0, 12.0]
key_redshifts = [0, 2, 4, 6, 8, 10, 12]

for model in models_to_try:
    print(f"\n--- {model} ---")
    header = f"{'log M':>8s}" + "".join([f"  z={z:<4d}" for z in key_redshifts])
    print(header)
    for lm in key_masses_log:
        M = 10**lm
        row = f"{lm:>8.1f}"
        for z in key_redshifts:
            try:
                c = concentration.concentration(M, 'vir', z, model=model)
                row += f"  {c:6.2f}"
            except:
                row += f"  {'N/A':>6s}"
        print(row)

print("\n" + "="*80)
print("Target: c ~ 4 at M = 10^10.8 Msun, z = 10")
print("="*80)

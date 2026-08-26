#!/usr/bin/env python
"""
feedback_scalings.py -- the SN feedback scalings actually implemented in
src/model_starformation_and_feedback.c, as functions of V_vir and redshift.

Produces the figure requested by the referee: the mass-loading factor
eta_reheat(V_vir, z) and the ejected-to-formed mass ratio
mdot_eject/mdot_* (V_vir, z), with the mass loadings realised by the
fiducial Millennium run overplotted as a sanity check.

The curves are the code expressions, not the printed equations:

    f_FIRE   = (1+z)^alpha * (V_vir / 60 km/s)^beta,  beta = -3.2 (V<60), -1 (V>=60)
    eta_reheat = eps_disk * f_FIRE                          [reheating, Muratov+15]
    E_FB     = eps_halo * f_FIRE * 0.5 * m_* * eta_SN E_SN  [ejection energy]
    E_lift   = 0.5 * eta_reheat * m_* * V_vir^2
    m_eject  = max(E_FB - E_lift, 0) / (0.5 V_vir^2)
             = eta_reheat * m_* * max(R - 1, 0),
      with   R = E_FB / E_lift = eps_halo eta_SN E_SN / (eps_disk V_vir^2)

Note that f_FIRE cancels in R: the ejection *threshold* is a pure function of
V_vir, independent of redshift, and sits at V_vir = 161 km/s for the fiducial
parameters.

Usage:
    python plotting/feedback_scalings.py [output_dir]

Reads:  <output_dir>/model_*.hdf5   (optional; for the realised-eta overlay)
Writes: <output_dir>/FeedbackScalings.pdf
"""

import glob
import os
import sys

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

# --- code constants (src/model_starformation_and_feedback.c) ---
V_CRIT_KMS = 60.0        # FIRE_V_CRIT_KMS
BETA_LOW = -3.2          # Muratov+15 slope below V_crit
BETA_HIGH = -1.0         # Muratov+15 slope above V_crit
MSUN_G = 1.989e33

REDSHIFTS = [0.0, 1.0, 2.0, 4.0, 6.0]


def sn_energy_per_mass_kms2(eta_sn, energy_sn):
    """eta_SN * E_SN expressed in (km/s)^2 per unit mass -- the combination
    the code carries as EtaSNcode * EnergySNcode (the Hubble_h factors cancel)."""
    return eta_sn * energy_sn / MSUN_G / 1.0e10


def fire_scaling(vvir, z, alpha):
    v = np.maximum(np.asarray(vvir, dtype=float), 1.0)
    beta = np.where(v < V_CRIT_KMS, BETA_LOW, BETA_HIGH)
    return (1.0 + z) ** alpha * (v / V_CRIT_KMS) ** beta


def eta_reheat(vvir, z, eps_disk, alpha):
    return eps_disk * fire_scaling(vvir, z, alpha)


def energy_ratio(vvir, eps_disk, eps_halo, esn_kms2):
    """E_FB / E_lift.  The FIRE scaling cancels, so this is z-independent."""
    v = np.asarray(vvir, dtype=float)
    return eps_halo * esn_kms2 / (eps_disk * v ** 2)


def eject_per_star(vvir, z, eps_disk, eps_halo, alpha, esn_kms2):
    r = energy_ratio(vvir, eps_disk, eps_halo, esn_kms2)
    return eta_reheat(vvir, z, eps_disk, alpha) * np.maximum(r - 1.0, 0.0)


def read_params(output_dir):
    """Feedback parameters from the run header, falling back to the shipped
    fiducial values when no output is available."""
    fallback = dict(eps_disk=2.9, eps_halo=0.3, alpha=1.25,
                    eta_sn=5.0e-3, energy_sn=1.0e51, hubble_h=0.73)
    files = sorted(glob.glob(os.path.join(output_dir, 'model_*.hdf5')))
    if not files:
        print(f'  No model files in {output_dir}; using shipped fiducial parameters.')
        return fallback, []
    with h5.File(files[0], 'r') as f:
        r = dict(f['Header/Runtime'].attrs)
        s = dict(f['Header/Simulation'].attrs)
    return dict(eps_disk=float(r['FeedbackReheatingEpsilon']),
                eps_halo=float(r['FeedbackEjectionEfficiency']),
                alpha=float(r['RedshiftPowerLawExponent']),
                eta_sn=float(r['EtaSN']),
                energy_sn=float(r['EnergySN']),
                hubble_h=float(s['hubble_h'])), files


def realised_mass_loading(files, z_targets):
    """Median MassLoading in V_vir bins for star-forming galaxies, per redshift.

    Returns {z_actual: (vvir_centres, median_eta)}.
    """
    if not files:
        return {}
    with h5.File(files[0], 'r') as f:
        zz = np.array(f['Header/snapshot_redshifts'])
    vbins = np.logspace(np.log10(20.0), np.log10(400.0), 16)
    out = {}
    for zt in z_targets:
        snap = int(np.argmin(np.abs(zz - zt)))
        eta, vv = [], []
        for fn in files:
            with h5.File(fn, 'r') as f:
                g = f[f'Snap_{snap}']
                if g['StellarMass'].shape[0] == 0:
                    continue
                sfr = np.array(g['SfrDisk']) + np.array(g['SfrBulge'])
                w = (sfr > 0) & (np.array(g['MassLoading']) > 0)
                eta.append(np.array(g['MassLoading'])[w])
                vv.append(np.array(g['Vvir'])[w])
        if not eta:
            continue
        eta = np.concatenate(eta)
        vv = np.concatenate(vv)
        cen, med = [], []
        for lo, hi in zip(vbins[:-1], vbins[1:]):
            m = (vv >= lo) & (vv < hi)
            if m.sum() >= 50:
                cen.append(np.sqrt(lo * hi))
                med.append(np.median(eta[m]))
        if cen:
            out[float(zz[snap])] = (np.array(cen), np.array(med))
    return out


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './output/millennium/'
    stylesheet = './plotting/kieren_cohare_palatino_sty.mplstyle'
    if os.path.exists(stylesheet):
        plt.style.use(stylesheet)

    p, files = read_params(output_dir)
    esn = sn_energy_per_mass_kms2(p['eta_sn'], p['energy_sn'])
    v_eject = np.sqrt(p['eps_halo'] * esn / p['eps_disk'])

    print(f"  eps_disk = {p['eps_disk']}, eps_halo = {p['eps_halo']}, "
          f"alpha_z = {p['alpha']}, eta_SN E_SN = {p['eta_sn']*p['energy_sn']:.2e} erg/Msun")
    print(f"  eta_SN E_SN = {esn:.4e} (km/s)^2  ->  V_SN = sqrt(eta_SN E_SN) = {np.sqrt(esn):.0f} km/s")
    print(f"  ejection threshold  E_FB = E_lift  at V_vir = {v_eject:.1f} km/s (all z)")

    vvir = np.logspace(np.log10(10.0), np.log10(500.0), 400)
    realised = realised_mass_loading(files, REDSHIFTS)

    cmap = plt.get_cmap('viridis')
    colours = [cmap(x) for x in np.linspace(0.05, 0.85, len(REDSHIFTS))]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    ax1, ax2 = axes

    # ---- panel (a): mass loading ----
    for z, c in zip(REDSHIFTS, colours):
        ax1.plot(vvir, eta_reheat(vvir, z, p['eps_disk'], p['alpha']),
                 color=c, lw=2.5, label=rf'$z = {z:.0f}$')
    for (z_act, (cen, med)), c in zip(sorted(realised.items()), colours):
        ax1.plot(cen, med, 'o', ms=5, color=c, mec='k', mew=0.6, zorder=5)

    ax1.axvline(V_CRIT_KMS, color='0.5', ls=':', lw=1.4)
    ax1.text(V_CRIT_KMS * 1.05, 3e3, r'$V_{\rm crit} = 60\ {\rm km\,s^{-1}}$',
             color='0.4', fontsize=11, rotation=90, va='top')
    ax1.axhline(1.0, color='0.7', ls='-', lw=0.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(10, 500)
    ax1.set_ylim(0.1, 1e4)
    ax1.set_xlabel(r'$V_{\rm vir}\ [{\rm km\,s^{-1}}]$')
    ax1.set_ylabel(r'$\eta_{\rm reheat} = \dot{m}_{\rm reheat} / \dot{m}_{*}$')
    ax1.set_title('(a) mass loading', loc='left')
    ax1.legend(frameon=False, loc='upper right', ncol=1)
    ax1.text(0.03, 0.05,
             'lines: code expression\nmarkers: run medians',
             transform=ax1.transAxes, fontsize=10, color='0.35', va='bottom')

    # ---- panel (b): ejection ----
    for z, c in zip(REDSHIFTS, colours):
        y = eject_per_star(vvir, z, p['eps_disk'], p['eps_halo'], p['alpha'], esn)
        ax2.plot(vvir, np.where(y > 0, y, np.nan), color=c, lw=2.5,
                 label=rf'$z = {z:.0f}$')

    ax2.axvline(v_eject, color='crimson', ls='--', lw=1.6)
    ax2.text(v_eject * 0.93, 3e3,
             rf'$E_{{\rm FB}} = E_{{\rm lift}}$ at ${v_eject:.0f}\ {{\rm km\,s^{{-1}}}}$'
             '\n(independent of $z$)',
             color='crimson', fontsize=11, rotation=90, va='top', ha='right')
    ax2.axhline(1.0, color='0.7', ls='-', lw=0.8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(10, 500)
    ax2.set_ylim(1e-2, 1e4)
    ax2.set_xlabel(r'$V_{\rm vir}\ [{\rm km\,s^{-1}}]$')
    ax2.set_ylabel(r'$\dot{m}_{\rm eject} / \dot{m}_{*}$')
    ax2.set_title('(b) ejection', loc='left')
    ax2.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    out = os.path.join(output_dir, 'FeedbackScalings.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f'  Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

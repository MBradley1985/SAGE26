#!/usr/bin/env python3
"""
Diagnostic: CGMPrecipRadiusMode=0 (r_cool) vs Mode=1 (0.1 R_vir)

Shows analytically why mode=0 gives near-maximum precipitation at all masses
while mode=1 correctly suppresses it in thermally stable massive haloes.

Two panels:
  Top    — t_cool/t_ff at the evaluation radius vs halo mass
  Bottom — resulting precipitation fraction vs halo mass
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import os

# ── Style ─────────────────────────────────────────────────────────────────────
_sty = os.path.join(os.path.dirname(__file__), '../../plotting/kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(_sty):
    plt.style.use(_sty)

# ── Physical constants (CGS) ──────────────────────────────────────────────────
G          = 6.674e-8
SOLAR_MASS = 1.989e33
MPC        = 3.0857e24
PROTON     = 1.6726e-24
BOLTZMANN  = 1.3806e-16
mu         = 0.6

# ── Cosmology (Millennium) ────────────────────────────────────────────────────
H0_kms    = 73.0
Omega_m   = 0.25
Omega_L   = 0.75
f_bar     = 0.17
f_CGM     = 0.5 * f_bar   # fiducial CGM gas fraction

H0_cgs = H0_kms * 1e5 / MPC   # s⁻¹

# ── Cooling function (Sutherland & Dopita, stripped_m-00.cie) ─────────────────
_cool_file = os.path.join(os.path.dirname(__file__),
                          '../../src/auxdata/CoolFunctions/stripped_m-00.cie')
_cool_data  = np.loadtxt(_cool_file)
_logT_table = _cool_data[:, 0]        # log10(T / K)
_logL_table = _cool_data[:, 4]        # log10(Lambda / erg cm³ s⁻¹)
_cool_interp = interp1d(_logT_table, _logL_table,
                        bounds_error=False, fill_value=-40.0)

def cooling_function(T_K):
    """Metal-free cooling function Lambda(T) in erg cm³ s⁻¹."""
    return 10.0 ** _cool_interp(np.log10(T_K))

# ── Cosmological helpers ──────────────────────────────────────────────────────
def E(z):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

def rho_crit_cgs(z):
    Hz = H0_cgs * E(z)
    return 3 * Hz**2 / (8 * np.pi * G)

def R_vir_cgs(Mvir_Msun, z, delta=200):
    M = Mvir_Msun * SOLAR_MASS
    return (3 * M / (4 * np.pi * delta * rho_crit_cgs(z)))**(1 / 3)

def V_vir_cgs(Mvir_Msun, z):
    M = Mvir_Msun * SOLAR_MASS
    R = R_vir_cgs(Mvir_Msun, z)
    return np.sqrt(G * M / R)

def T_vir_K(Mvir_Msun, z):
    Vv = V_vir_cgs(Mvir_Msun, z)
    return mu * PROTON * Vv**2 / (2 * BOLTZMANN)

def concentration(Mvir_Msun, z):
    """Duffy+08 mean concentration."""
    return 6.71 * (Mvir_Msun / 2e12)**(-0.091) * (1 + z)**(-0.44)

# ── NFW helpers ───────────────────────────────────────────────────────────────
def nfw_density(r, Mcgm, Rvir, c):
    rs   = Rvir / c
    fc   = np.log(1 + c) - c / (1 + c)
    rhos = Mcgm / (4 * np.pi * rs**3 * fc)
    x    = r / rs
    return rhos / (x * (1 + x)**2)

def nfw_enclosed(r, Mvir, Rvir, c):
    fc = np.log(1 + c) - c / (1 + c)
    x  = r / Rvir
    fx = np.log(1 + c * x) - c * x / (1 + c * x)
    return Mvir * fx / fc

# ── t_cool / t_ff at a fixed radius fraction ──────────────────────────────────
def ratio_at_r(r_frac, Mvir_Msun, z):
    """t_cool / t_ff at radius r_frac * R_vir, using NFW profile."""
    M    = Mvir_Msun * SOLAR_MASS
    Mcgm = f_CGM * M
    Rv   = R_vir_cgs(Mvir_Msun, z)
    c    = concentration(Mvir_Msun, z)
    T    = T_vir_K(Mvir_Msun, z)
    lam  = cooling_function(T)

    r    = r_frac * Rv
    rho  = nfw_density(r, Mcgm, Rv, c)
    Menc = nfw_enclosed(r, M, Rv, c)

    if lam <= 0 or rho <= 0 or Menc <= 0:
        return np.inf

    tcool = 1.5 * mu * PROTON * BOLTZMANN * T / (rho * lam)
    g     = G * Menc / r**2
    tff   = np.sqrt(2 * r / g)
    return tcool / tff

# ── Precipitation fraction (sigmoid, matching SAGE26 code) ───────────────────
THRESHOLD   = 10.0
DELTA_WIDTH = 2.0

def precip_fraction(ratio):
    x = (THRESHOLD - np.asarray(ratio, dtype=float)) / DELTA_WIDTH
    return 1.0 / (1.0 + np.exp(-x))

# ── Main ──────────────────────────────────────────────────────────────────────
M_arr     = np.logspace(10, 14, 120)   # Msun
redshifts = [2, 5, 8]
colors    = ['steelblue', 'darkorange', 'firebrick']
M_shock   = 6e11   # D&B06 virial shock threshold [Msun]

fig, (ax_r, ax_f) = plt.subplots(2, 1, figsize=(6.5, 8), sharex=True)

for z, col in zip(redshifts, colors):
    ratio_01 = np.array([ratio_at_r(0.1, M, z) for M in M_arr])

    # Mode 1: 0.1 R_vir
    ax_r.plot(M_arr, ratio_01, color=col, lw=2)
    ax_f.plot(M_arr, precip_fraction(ratio_01), color=col, lw=2, label=f'$z = {z}$')

    # Mode 0: r_cool → ratio ≡ 1
    ax_r.axhline(1.0, color=col, lw=1.5, ls='--', alpha=0.7)
    ax_f.axhline(precip_fraction(1.0), color=col, lw=1.5, ls='--', alpha=0.7)

# Threshold line
ax_r.axhline(THRESHOLD, color='k', lw=1, ls=':', zorder=5)
ax_r.text(1.05e10, THRESHOLD * 1.15, r'threshold $= 10$',
          fontsize=9, va='bottom')

# M_shock marker
for ax in (ax_r, ax_f):
    ax.axvline(M_shock, color='grey', lw=1, ls=':', alpha=0.6)
ax_r.text(M_shock * 1.1, 3e3, r'$M_{\rm shock}$',
          fontsize=9, color='grey', va='top')

# Shading: stable / unstable
ax_r.axhspan(0.3, THRESHOLD, alpha=0.04, color='red')
ax_r.axhspan(THRESHOLD, 5e4, alpha=0.04, color='steelblue')
ax_r.text(1.05e10, 1.3, 'Unstable', fontsize=8, color='firebrick', va='bottom')
ax_r.text(1.05e10, THRESHOLD * 2.5, 'Stable', fontsize=8, color='steelblue', va='bottom')

# ── Labels ────────────────────────────────────────────────────────────────────
ax_r.set_ylabel(r'$t_{\rm cool} / t_{\rm ff}$ at evaluation radius')
ax_r.set_yscale('log')
ax_r.set_ylim(0.3, 5e4)

ax_f.set_ylabel(r'Precipitation fraction $f_{\rm precip}$')
ax_f.set_xlabel(r'$M_{\rm vir}\ [M_\odot]$')
ax_f.set_ylim(-0.02, 1.05)

ax_r.set_xscale('log')
ax_r.set_xlim(M_arr[0], M_arr[-1])

# ── Legend ────────────────────────────────────────────────────────────────────
style_lines = [
    Line2D([0], [0], color='k', lw=2,   ls='-',  label=r'Mode 1: $0.1\,R_{\rm vir}$'),
    Line2D([0], [0], color='k', lw=1.5, ls='--', label=r'Mode 0: $r_{\rm cool}\ (\equiv 1)$'),
]
z_lines = [Line2D([0], [0], color=c, lw=2, label=f'$z = {z}$')
           for z, c in zip(redshifts, colors)]
ax_f.legend(handles=style_lines + z_lines, fontsize=9, loc='center right')

fig.suptitle('Precipitation criterion: radius evaluation mode', fontsize=11)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), '../../plotting/precip_radius_mode.pdf')
plt.savefig(out, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved {out}')
plt.close()

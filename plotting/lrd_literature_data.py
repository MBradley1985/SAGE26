"""
lrd_literature_data.py
=======================
Hand-transcribed observational LRD/broad-line-AGN tables for overlay onto
the panels in bh_lrd_analysis.py. Each entry below cites the table/figure it
was taken from; values are used as given except where a panel's axis
requires a quantity the source table doesn't tabulate directly, in which
case it is derived using the SAME physical assumptions (eta=0.1, L_Edd =
1.3e38 M_BH/Msun erg/s) as the rest of bh_lrd_analysis.py -- see that
module's mdot_from_lbol / bh_mass_min_from_lbol / eddington_luminosity.

All masses/luminosities are already in the units bh_lrd_analysis.py plots
(log10(Msun), log10(erg/s), linear Eddington ratio, absolute AB magnitude).
"""

import numpy as np

# ============================================================================
# Pang et al. (2026) -- Table 1, "LRD Properties & Evolution"
# Columns used: Redshift, log M_BH, lambda_Edd, log M_star (all with quoted
# 1-sigma errors already in dex / linear as appropriate).
# ============================================================================
PANG26 = {
    'label': 'Pang+26',
    'z':            np.array([3.94, 3.69, 2.26, 4.22, 4.13, 3.43, 4.13, 4.15, 3.35, 3.55, 4.12]),
    'log_mbh':      np.array([6.41, 6.70, 7.29, 6.95, 6.23, 6.56, 6.58, 6.81, 6.90, 6.49, 6.99]),
    'log_mbh_err':  np.array([0.04, 0.11, 0.01, 0.13, 0.03, 0.12, 0.09, 0.07, 0.07, 0.21, 0.10]),
    'lambda_edd':     np.array([0.36, 0.35, 0.84, 0.56, 1.55, 0.73, 1.21, 0.30, 0.31, 2.08, 0.18]),
    'lambda_edd_err': np.array([0.04, 0.09, 0.03, 0.25, 0.11, 0.20, 0.24, 0.05, 0.06, 1.07, 0.05]),
    'log_mstar':     np.array([9.32, 9.16, 9.54, 8.70, 8.53, 9.15, 9.44, 8.73, 8.59, 8.42, 9.53]),
    'log_mstar_err': np.array([0.10, 0.13, 0.04, 0.10, 0.15, 0.07, 0.07, 0.08, 0.09, 0.10, 0.18]),
}

# ============================================================================
# Mathee et al. (2024) -- Table 3, "BH and Galaxy Properties of the BL Halpha
# Emitters". L_bol tabulated in units of 1e44 erg/s; M_UV given directly.
# ============================================================================
MATHEE24 = {
    'label': 'Mathee+24',
    'id': ['GOODS-N-4014', 'GOODS-N-9771', 'GOODS-N-12839', 'GOODS-N-13733',
           'GOODS-N-14409', 'GOODS-N-15498', 'GOODS-N-16813', 'GOODS-S-13971',
           'J1148-7111', 'J1148-18404', 'J1148-21787', 'J0100-2017',
           'J0100-12446', 'J0100-15157', 'J0100-16221', 'J0148-976',
           'J0148-4214', 'J0148-12884', 'J1120-7546', 'J1120-14389'],
    'log_mbh':     np.array([7.58, 8.55, 8.01, 7.49, 7.21, 7.71, 7.55, 7.49,
                              7.92, 7.79, 7.59, 7.44, 7.46, 7.35, 7.53, 7.11,
                              7.32, 6.91, 7.56, 7.65]),
    'log_mbh_err': np.array([0.08, 0.03, 0.06, 0.10, 0.14, 0.11, 0.12, 0.25,
                              0.10, 0.14, 0.18, 0.11, 0.06, 0.08, 0.07, 0.18,
                              0.10, 0.15, 0.11, 0.07]),
    'lbol_1e44':     np.array([9.3, 65.8, 31.2, 5.2, 7.4, 10.4, 9.1, 5.5,
                                10.8, 6.9, 6.7, 6.7, 12.9, 6.5, 7.1, 5.2,
                                5.9, 5.0, 13.8, 8.4]),
    'lbol_1e44_err': np.array([0.5, 1.6, 1.2, 0.3, 0.9, 1.9, 1.0, 1.2,
                                0.8, 1.4, 1.6, 0.8, 0.6, 0.6, 0.6, 0.7,
                                0.4, 0.6, 1.6, 0.8]),
    'muv':     np.array([-18.0, -19.5, -19.0, -17.9, -18.3, -17.7, -19.7, -19.4,
                          -18.6, -17.5, -19.2, -19.3, -19.0, -20.2, -18.9, -19.1,
                          -20.0, -19.5, -17.6, -19.1]),
    'muv_err': np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.3, 0.1]),
}

# ============================================================================
# Labbe et al. (2025) -- Table 3, "SED Fits". No M_BH is tabulated -- only
# L_bol (AGN-only fit) and M_1450, for SED == 1 rows (SED-classified AGN
# sample; col 12). Used ONLY to derive a lower-limit M_BH assuming
# Eddington-limited accretion (M_BH_min = L_bol / L_Edd_per_Msun) -- see
# bh_mass_min_from_lbol() in bh_lrd_analysis.py. No errors are tabulated for
# log(L_bol) or M_1450 in the source table.
# ============================================================================
LABBE25 = {
    'label': 'Labbe+25',
    'id':          [571, 2940, 5957, 6430, 8296, 8798, 9992, 10148, 10712,
                     13556, 16561, 20080, 21860, 23778, 28343, 29466, 30782],
    'log_lbol':    np.array([44.4, 43.8, 45.0, 45.5, 46.2, 45.5, 45.9, 45.3, 45.7,
                              45.2, 44.7, 44.2, 42.9, 45.3, 44.9, 46.2, 45.3]),
    'm1450':       np.array([-17.6, -16.4, -17.5, -18.1, -16.5, -18.0, -16.5, -16.9, -16.0,
                              -16.9, -16.4, -16.9, -14.5, -18.0, -18.7, -18.2, -17.9]),
}

# ============================================================================
# Furtak et al. (2023) -- Figure only (panels a-d), no tabulated values
# available. Values below are READ OFF THE FIGURE (approximate) for the
# single lensed z=7.04 "This work" point, using the panels' own gridlines:
#   panel a: M_UV,1450 vs log M_BH
#   panel b: L_bol    vs log M_BH
#   panel c: log M_star vs log M_BH (M_star shown as an upper limit, left arrow)
#   panel d: BH mass function point (phi vs log M_BH) -- see allresults-blackholes.py
# Treat these as illustrative, not precision measurements.
# ============================================================================
FURTAK23 = {
    'label': 'Furtak+23',
    'z': 7.04,
    'log_mbh': 7.48, 'log_mbh_err_lo': 0.48, 'log_mbh_err_hi': 0.50,
    'muv': -15.2, 'muv_err': 0.3,
    'log_lbol': 45.05, 'log_lbol_err': 0.35,
    'log_mstar_upper_limit': 9.15,
    'log_phi_bhmf': -3.7,   # Mpc^-3 dex^-1, panel (d) of the figure
}

# ============================================================================
# Lin et al. (2025) -- small BL-Halpha sample table. log M_BH, log L_bol,Ha,
# and lambda_Edd,Ha all tabulated directly.
# ============================================================================
LIN25 = {
    'label': 'Lin+25',
    'id': ['J1025+1402', 'J1047+0739', 'J1022+0841'],
    'log_mbh':        np.array([6.52, 6.76, 6.74]),
    'log_mbh_err':    np.array([0.03, 0.02, 0.04]),
    'log_lbol':       np.array([44.03, 44.55, 44.83]),
    'log_lbol_err':   np.array([0.01, 0.01, 0.01]),
    'lambda_edd':     np.array([0.21, 0.41, 0.81]),
    'lambda_edd_err': np.array([0.01, 0.02, 0.07]),
}

# ============================================================================
# Shen et al. (2020, MNRAS 495, 3252; arXiv:2001.02696) -- bolometric quasar
# luminosity function, "global fit A" (their Eq. 11 & 14, Table 4). This is
# a continuous model, not a points table: phi_bol(L, z) = dn/dlogL is a
# double power-law in L with redshift-dependent break luminosity, faint/
# bright-end slopes, and normalization. See shen20_bolometric_qlf_logphi()
# in bh_lrd_analysis.py for the evaluator built from these coefficients.
#
# log L_star is tabulated by Shen+20 in L_sun; phi_star in dex^-1 cMpc^-3
# for their assumed H0 = 70 km/s/Mpc (i.e. NOT h-scaled -- see the h^3
# conversion applied where this is plotted, to match this code's own
# dex^-1 (Mpc/h)^-3 convention for a different simulation h).
# ============================================================================
SHEN20_ZREF = 2.0
SHEN20_GAMMA1_A  = (0.8569, -0.2614, 0.0200)     # a0, a1, a2
SHEN20_GAMMA2_A  = (2.5375, -1.0425, 1.1201)     # b0, b1, b2
SHEN20_LSTAR_A   = (13.0088, -0.5759, 0.4554)    # c0, c1, c2 -- log L_star [Lsun]
SHEN20_PHISTAR_A = (-3.5426, -0.3936)            # d0, d1     -- log phi_star [dex^-1 cMpc^-3]

# ── shared marker/colour style so the same source looks the same across panels ──
LIT_STYLE = {
    'Pang+26': {'marker': 'D', 'color': '#8E24AA', 'ms': 7},
    'Mathee+24':  {'marker': 's', 'color': '#43A047', 'ms': 7},
    'Labbe+25':   {'marker': '^', 'color': '#1E88E5', 'ms': 7},
    'Furtak+23':  {'marker': '*', 'color': '#FB8C00', 'ms': 15},
    'Lin+25':     {'marker': 'v', 'color': '#6D4C41', 'ms': 8},
    'Shen+20':    {'marker': None, 'color': '#000000', 'ms': 0},
}

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
    # Table 1 (coordinates & redshifts) of Matthee et al. (2024, ApJ 963, 129)
    'z':           np.array([5.228, 5.538, 5.241, 5.236, 5.139, 5.086, 5.355, 5.481,
                              4.339, 5.011, 4.277, 4.938, 4.699, 4.941, 4.349, 4.163,
                              5.019, 4.602, 4.967, 4.897]),
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
    # Table 3 (SED fits) photometric redshifts, Labbe et al. (2025, ApJ 978, 92)
    'z':           np.array([5.50, 4.47, 3.52, 4.98, 6.80, 6.44, 6.79, 4.96, 6.77,
                              5.17, 6.34, 5.52, 5.46, 7.02, 4.95, 7.22, 6.74]),
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
    # Table 1 of Lin et al. (2025, arXiv:2507.10659) -- these are LOCAL LRD
    # analogues at z~0.1-0.2, not high-redshift sources.
    'z':              np.array([0.1007, 0.1682, 0.2227]),
    'log_mbh':        np.array([6.52, 6.76, 6.74]),
    'log_mbh_err':    np.array([0.03, 0.02, 0.04]),
    'log_lbol':       np.array([44.03, 44.55, 44.83]),
    'log_lbol_err':   np.array([0.01, 0.01, 0.01]),
    'lambda_edd':     np.array([0.21, 0.41, 0.81]),
    'lambda_edd_err': np.array([0.01, 0.02, 0.07]),
}

# ============================================================================
# Table 1 of Jones et al. (2025, arXiv:2510.07376), "Properties of Our Sample
# of Broad-line AGN", compiles the M_BH/M_star measurements of five separate
# JWST NIRSpec broad-line AGN studies -- it is a compilation, not an original
# measurement, so each LRD == 1 row below (heavily-reddened, "v-shaped" SED
# sources) is attributed to the study that actually measured it, per Table 1's
# own per-source grouping. log(M_BH) is quoted there with asymmetric errors
# (single-epoch viral estimate from broad Halpha); log(M_star) errors are
# symmetric. No L_bol or M_UV is tabulated for these sources, so they appear
# only in the M_BH-M_star and f_BH panels (unlike PANG26 they can't be shown
# on the Lbol/Mdot panels).
# ============================================================================
KOCEVSKI23 = {
    'label': 'Kocevski+23',
    'id': ['CEERS 746'],
    'z':             np.array([5.624]),
    'log_mbh':       np.array([7.19]),
    'log_mbh_err_lo': np.array([0.13]),
    'log_mbh_err_hi': np.array([0.11]),
    'log_mstar':     np.array([7.99]),
    'log_mstar_err': np.array([0.37]),
}

HARIKANE23 = {
    'label': 'Harikane+23',
    'id': ['CEERS 672'],
    'z':             np.array([5.666]),
    'log_mbh':       np.array([6.93]),
    'log_mbh_err_lo': np.array([0.14]),
    'log_mbh_err_hi': np.array([0.13]),
    'log_mstar':     np.array([8.33]),
    'log_mstar_err': np.array([0.24]),
}

MAIOLINO23 = {
    'label': 'Maiolino+23',
    'id': ['JADES-GN 954', 'JADES-GN 1093', 'JADES-GN 61888'],
    'z':             np.array([6.760, 5.595, 5.875]),
    'log_mbh':       np.array([7.60, 6.81, 6.86]),
    'log_mbh_err_lo': np.array([0.07, 0.21, 0.11]),
    'log_mbh_err_hi': np.array([0.07, 0.18, 0.10]),
    'log_mstar':     np.array([8.92, 7.82, 8.24]),
    'log_mstar_err': np.array([0.76, 0.32, 0.32]),
}

KOCEVSKI25 = {
    'label': 'Kocevski+25',
    'id': ['RUBIES-EGS 37124', 'RUBIES-EGS 42046', 'RUBIES-EGS 42232', 'RUBIES-EGS 49140',
           'RUBIES-EGS 55604', 'RUBIES-EGS 60935', 'RUBIES-EGS 61496', 'RUBIES-EGS 926125',
           'RUBIES-EGS 927271', 'RUBIES-UDS 40579', 'RUBIES-UDS 50716', 'RUBIES-UDS 59971'],
    'z': np.array([5.684, 5.280, 4.955, 6.687,
                   6.986, 5.288, 5.079, 5.286,
                   6.786, 3.103, 6.17, 5.365]),
    'log_mbh': np.array([7.18, 8.47, 7.58, 8.72,
                         8.55, 7.39, 7.00, 7.10,
                         6.74, 8.29, 7.26, 6.74]),
    'log_mbh_err_lo': np.array([0.08, 0.02, 0.03, 0.09,
                                0.12, 0.04, 0.18, 0.04,
                                0.18, 0.06, 0.12, 0.21]),
    'log_mbh_err_hi': np.array([0.07, 0.02, 0.03, 0.08,
                                0.11, 0.04, 0.15, 0.04,
                                0.16, 0.05, 0.10, 0.18]),
    'log_mstar': np.array([8.42, 8.96, 8.71, 9.01,
                           9.05, 8.49, 8.68, 8.32,
                           8.70, 8.09, 8.16, 8.61]),
    'log_mstar_err': np.array([0.37, 0.46, 0.46, 0.35,
                               0.52, 0.29, 0.46, 0.36,
                               0.36, 0.51, 0.43, 0.48]),
}

TAYLOR25 = {
    'label': 'Taylor+25',
    'id': ['RUBIES-EGS 28812', 'RUBIES-EGS 29489', 'RUBIES-EGS 37032', 'RUBIES-EGS 50812',
           'RUBIES-UDS 19521', 'RUBIES-UDS 29813', 'RUBIES-UDS 119957', 'RUBIES-UDS 139709',
           'RUBIES-UDS 150323', 'RUBIES-UDS 172350', 'RUBIES-UDS 182791', 'RUBIES-UDS 807469'],
    'z': np.array([4.223, 4.543, 3.850, 3.519,
                   5.669, 5.440, 4.149, 5.685,
                   3.618, 5.580, 4.718, 6.778]),
    'log_mbh': np.array([7.37, 7.56, 7.28, 7.17,
                         7.03, 7.52, 7.22, 7.86,
                         7.12, 7.63, 7.93, 7.19]),
    'log_mbh_err_lo': np.array([0.03, 0.08, 0.09, 0.15,
                                0.14, 0.08, 0.05, 0.06,
                                0.06, 0.04, 0.02, 0.14]),
    'log_mbh_err_hi': np.array([0.03, 0.08, 0.08, 0.13,
                                0.13, 0.07, 0.05, 0.06,
                                0.06, 0.04, 0.02, 0.12]),
    'log_mstar': np.array([9.01, 8.65, 8.67, 8.22,
                           8.14, 8.29, 8.07, 8.65,
                           8.94, 8.23, 9.00, 8.34]),
    'log_mstar_err': np.array([0.43, 0.39, 0.33, 0.38,
                               0.92, 0.35, 0.45, 0.41,
                               0.54, 0.47, 0.58, 0.53]),
}

# Convenience grouping -- all five sources share the exact same schema (z,
# log_mbh[_err_lo/_hi], log_mstar[_err]), so panels that overlay "the LRD == 1
# rows of Jones+25 Table 1" can just iterate this list instead of naming each
# study individually.
JONES25_LRD_SOURCES = [KOCEVSKI23, HARIKANE23, MAIOLINO23, KOCEVSKI25, TAYLOR25]

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
    'Kocevski+23': {'marker': 'p', 'color': '#00ACC1', 'ms': 8},
    'Harikane+23': {'marker': 'h', 'color': '#7CB342', 'ms': 8},
    'Maiolino+23': {'marker': 'X', 'color': '#3949AB', 'ms': 8},
    'Kocevski+25': {'marker': '<', 'color': '#D81B60', 'ms': 8},
    'Taylor+25':   {'marker': '>', 'color': '#F4511E', 'ms': 8},
    'Shen+20':    {'marker': None, 'color': '#000000', 'ms': 0},
}

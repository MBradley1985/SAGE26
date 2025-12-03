"""
Feedback-Free Starbursts at Cosmic Dawn: Observable Predictions for JWST
Zhaozhou Li, Avishai Dekel, Kartick C. Sarkar, Han Aung, Mauro Giavalisco, Nir Mandelker, and Sandro Tacchella


author: Zhaozhou Li (lizz.astro@gmail.com)
"""
from contextlib import contextmanager

import numpy as np
from numpy import pi, log10, log, exp
from scipy.interpolate import Akima1DInterpolator
from scipy.special import erf, roots_legendre
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
# from h5attr import H5Attr

from colossus.cosmology import cosmology
from colossus.lss import mass_function, peaks


# cosmology
# -----------------------------
# params = {'flat': True, 'H0': 67.32, 'Om0': 0.3158, 'Ob0': 0.156 * 0.3158, 'sigma8': 0.812, 'ns': 0.96605}  # Planck15
params = {"flat": True, "H0": 70, "Om0": 0.3, "Ob0": 0.048, "sigma8": 0.82, "ns": 0.95}
cosmo = cosmology.setCosmology("myCosmo", params)
fb = cosmo.Ob0 / cosmo.Om0  # 0.16

# Press-Schechter parameters
ps_ext_args = dict(
    ps_args={"model": "eisenstein98"},
    sigma_args={"filt": "tophat"},
    deltac_args={"corrections": True},
)

# global variables
# -----------------------------
default_options = dict(
    UM_SHMR_MINSLOP=0.3,  # manual fix with minimum slop of dlnM*/dlnMh, None for disable
    FFB_SFR_SIG=0.3,  # dex
    FFB_SFR_TH_SIG=0.15,  # dex, transition width, equiv to ~0.6 dex full transition region
    FFB_LGMH_PIVOT=10.8,  # Msun
    FFB_LGMH_QUENCH=None,  # Msun, can be None to disable
    FFB_SFE_MAX=1,  # maximum star formation efficiency for FFB galaxies
    FFB_FRAC_ALL=False,  # if true, f_ffb is defined wrt all gals, otherwise wrt SF gals.
    HALO_MAH_MODEL="Dekel13",  # Model for halo growth history, Dekel13 or Zhao09
    DEKEL13_BETA=0.14,  # beta=0.14 in Dekel+13 eq. 7
    DEKEL13_PIVOT=12,  # pivot mass
    UV_FACTOR=1,  # not used
)
options = dict()


def set_option(**opts):
    for key, val in opts.items():
        if key in default_options:
            options[key] = val
        else:
            raise ValueError(f"{key} is not valid option.")
    return options


set_option(**default_options)


def get_option():
    return options.copy()


@contextmanager
def with_option(**opts):
    "change the options temporarily, restore to earlier values when exit the context"
    current_options = get_option()
    try:
        set_option(**opts)
        yield
    finally:
        set_option(**current_options)


# math functions
# -----------------------------
def sigmoid(x, m=0, s=1, a=0, b=1):
    """
    varying from a to b, with m the midpoint and s the width

    example:
        x = linspace(-5, 5, 100)
        plot(x, sigmoid(x, s=1))
        abline((0, 0.5), 0.25)
    """
    return exp(-np.logaddexp(0, -(x - m) / s)) * (b - a) + a


def normal(x, m, s):
    return exp(-0.5 * ((x - m) / s) ** 2) / (s * (2 * pi) ** 0.5)


def lgnormal_mean(med, sig_dex):
    return med * exp(0.5 * (sig_dex * log(10)) ** 2)


# Universe Machine functions
# -----------------------------
def um_vpeak_Mh_z(Mh, z):
    # peak historical halo mass, using Bryan & Norman virial overdensity.
    # B19, eq E2
    # Mh: Msun, vpeak: km/s
    a = 1.0 / (1.0 + z)
    Mz = 1.64e12 / ((a / 0.378) ** -0.142 + (a / 0.378) ** -1.79)
    vpeak = 200 * (Mh / Mz) ** (1 / 3)
    return vpeak


def um_Mh_vpeak_z(vpeak, z):
    # peak historical halo mass, using Bryan & Norman virial overdensity.
    # B19, eq E2
    # Mh: Msun, vpeak: km/s
    a = 1.0 / (1.0 + z)
    Mz = 1.64e12 / ((a / 0.378) ** -0.142 + (a / 0.378) ** -1.79)
    Mh = Mz * (vpeak / 200) ** 3
    return Mh


def _um_sfr_func1(z, x0, xa, xla, xz):
    a = 1.0 / (1.0 + z)
    return x0 + xa * (1 - a) + xla * log(1 + z) + xz * z


def _um_sfr_func2(z, x0, xa, xz):
    a = 1.0 / (1.0 + z)
    return x0 + xa * (1 - a) + xz * z


def um_sfr_med_sf(lgMh, z):
    # B19, appendix H
    vpeak = um_vpeak_Mh_z(10**lgMh, z)  # km/s
    vz = 10 ** _um_sfr_func1(z, 2.151, -1.658, 1.680, -0.233)  # km/s
    eps = 10 ** _um_sfr_func1(z, 0.109, -3.441, 5.079, -0.781)  # Msun/yr
    alpha = _um_sfr_func1(z, -5.598, -20.731, 13.455, -1.321)
    beta = _um_sfr_func2(z, -1.911, 0.395, 0.747)
    gamma = 10 ** _um_sfr_func2(z, -1.699, 4.206, -0.809)
    delta = 0.055

    # B19, eq 4, 5
    v = vpeak / vz
    SFR_SF = eps * (
        (v**alpha + v**beta) ** (-1) + gamma * exp(-0.5 * (log10(v) / delta) ** 2)
    )
    return SFR_SF


def um_sfr_med_q(lgMh, z):
    # XXX
    Mstar = 10 ** um_lgMs_med(lgMh, z)  # Msun
    SFR_Q = 10**-11.8 * Mstar  # Msun/yr
    return SFR_Q


def um_sfr_sig_sf(lgMh, z):
    # B19, eq 3, appendix H
    # a = 1.0 / (1.0 + z)
    # sig_sf = np.fmin(-4.361 + 26.926 * (1 - a), 0.3)  # dex
    # sig_sf = np.clip(sig_sf, 1e-5, None)  # minimum sig
    sig_sf = 0.3  # dex, the above formula is problematic
    return sig_sf


def um_sfr_sig_q(lgMh, z):
    sig_q = 0.36  # dex
    return sig_q


def um_lgMhQ(z):
    # B19, eq 12-15, appendix H
    a = 1.0 / (1.0 + z)
    vq = 10 ** (2.248 - 0.018 * (1 - a) + 0.124 * z)
    Mh = um_Mh_vpeak_z(vq, z)
    return log10(Mh)


def um_fQ(lgMh, z):
    a = 1.0 / (1.0 + z)
    vpeak = um_vpeak_Mh_z(10**lgMh, z)  # km/s

    # B19, eq 12-15, appendix H
    qmin = np.fmax(0, -1.944 - 2.419 * (1 - a))
    vq = 10 ** (2.248 - 0.018 * (1 - a) + 0.124 * z)
    sigvq = 0.227 + 0.037 * (1 - a) - 0.107 * log(1 + z)  # B19, eq 14
    # note the typo in appendix H
    sigvq = np.fmax(0.01, sigvq)  # ZZ: ensure sigvq>=0.01

    x = log10(vpeak / vq) / (2**0.5 * sigvq)
    f_Q = qmin + (1 - qmin) * (0.5 + 0.5 * erf(x))  # eq 12

    return f_Q


def um_sfr(lgMh, z):
    SFR_SF = um_sfr_med_sf(lgMh, z)
    SFR_Q = um_sfr_med_q(lgMh, z)
    f_Q = um_fQ(lgMh, z)
    return f_Q * SFR_Q + (1 - f_Q) * SFR_SF


def um_lgMs_med_basic(lgMh, z, model="obs"):
    # Mh: Msun, Ms: Msun
    # B19, appendix J
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = log(a)

    # fmt: off
    if model == 'obs':
        # B19, Table J1: Obs, All, All, Excl
        (EFF_0, EFF_A, EFF_A2, EFF_Z,
            M1_0, M1_A, M1_A2, M1_Z,
            ALPHA_0, ALPHA_A, ALPHA_A2, ALPHA_Z,
            BETA_0, BETA_A, BETA_Z, DELTA_0,
            GAMMA_0, GAMMA_A, GAMMA_Z) = (
            -1.4346, 1.8313, 1.3683, -0.2169,
            12.0354, 4.5562, 4.4171, -0.7314,
            1.9633, -2.3156, -1.7321, 0.1776,
            0.4818, -0.8406, -0.4707, 0.4109,
            -1.0342, -3.1004, -1.0545)
    elif model == 'true':
        # B19, Table J1: True, All, All, Excl
        (EFF_0, EFF_A, EFF_A2, EFF_Z,
            M1_0, M1_A, M1_A2, M1_Z,
            ALPHA_0, ALPHA_A, ALPHA_A2, ALPHA_Z,
            BETA_0, BETA_A, BETA_Z, DELTA_0,
            GAMMA_0, GAMMA_A, GAMMA_Z) = (
            -1.4305, 1.7958, 1.3596, -0.2156,
            12.0400, 4.6752, 4.5131, -0.7444,
            1.9731, -2.3534, -1.7833, 0.1860,
            0.4732, -0.8843, -0.4861, 0.4068,
            -1.0879, -3.2414, -1.0785)
    # fmt: on
    lgm1 = M1_0 + a1 * M1_A - lna * M1_A2 + z * M1_Z
    eff = EFF_0 + a1 * EFF_A - lna * EFF_A2 + z * EFF_Z
    alpha = ALPHA_0 + a1 * ALPHA_A - lna * ALPHA_A2 + z * ALPHA_Z
    beta = BETA_0 + a1 * BETA_A + z * BETA_Z
    delta = DELTA_0
    gamma = 10 ** (GAMMA_0 + a1 * GAMMA_A + z * GAMMA_Z)

    if options["UM_SHMR_MINSLOP"] is not None:
        beta = np.fmax(beta, options["UM_SHMR_MINSLOP"])  # slope for high mass end

    x = lgMh - lgm1
    lgMs = lgm1 + (
        eff
        - log10(10 ** (-alpha * x) + 10 ** (-beta * x))
        + gamma * exp(-0.5 * (x / delta) ** 2)
    )
    return lgMs  # Msun


def um_lgMs_sig(lgMh, z):
    """
    x = linspace(10, 15, 100)
    plot(x, sigmoid(x, 12.5, s=1, a=0.3, b=0.2))
    grid(ls='--')
    xlim(10, 15)
    ylim(0, 0.5)
    """
    # sig = sigmoid(lgMh, 12.5, s=1, a=0.3, b=0.2)  # B19, fig 12
    # sig = sigmoid(lgMh, 13.9 - z * 0.3, s=0.2, a=0.4, b=0.1)  # according to Han
    sig = sigmoid(lgMh, 13.9 - z * 0.3, s=0.2, a=0.4, b=0.3)
    # added scatter due to scatter in MUV(Ms)
    return sig  # dex


def um_lgMs_med(lgMh, z):
    "ensure dlgMs/dlgMh >= options['UM_SHMR_MINSLOP']"
    lgMs = um_lgMs_med_basic(lgMh, z)

    # if options['UM_SHMR_MINSLOP'] is not None:
    #     lgMh_ = np.linspace(0, 16, 161)
    #     lgMs_ = um_lgMs_med_basic(lgMh_, z=z)

    #     f = Akima1DInterpolator(lgMh_, lgMs_)
    #     roots = f.derivative().solve(options['UM_SHMR_MINSLOP'], extrapolate=False)
    #     if len(roots) == 0:
    #         lgMh_peak = lgMh_[-1]
    #     else:
    #         lgMh_peak = roots[0]
    #     lgMs_peak = f(lgMh_peak)

    #     if np.isscalar(lgMh):
    #         if lgMh > lgMh_peak:
    #             lgMs = lgMs_peak + options['UM_SHMR_MINSLOP'] * (lgMh - lgMh_peak)
    #     else:
    #         ix = lgMh > lgMh_peak
    #         lgMs[ix] = lgMs_peak + options['UM_SHMR_MINSLOP'] * (lgMh[ix] - lgMh_peak)

    return lgMs


def um_AUV(MUV, z, derivative=False):
    # B19, eq 23, 24
    M_dust = -20.594 - 0.054 * (np.fmax(4, z) - 4)
    alpha_dust = 0.559
    z = 10 ** (0.4 * alpha_dust * (M_dust - MUV))
    AUV = 2.5 * log10(1 + z)

    if derivative:
        dAUV_dMUV = -alpha_dust * z / (1 + z)
        return AUV, dAUV_dMUV
    else:
        return AUV


# FFB functions
# -----------------------------
def ffb_lgMh_crit(z):
    "Halo mass threshold for FFB"
    # lgMh_crit = 10.8 - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    lgMh_crit = options["FFB_LGMH_PIVOT"] - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit


def ffb_sfr_med(lgMh, z):
    sfr_avg = func_Mdot_baryon(lgMh, z=z) * options["FFB_SFE_MAX"]  # keep sfr_avg fixed
    fac_avg_med = exp(0.5 * (ffb_sfr_sig(lgMh, z=z) * log(10)) ** 2)
    sfr_med = sfr_avg / fac_avg_med  # convert avg to median
    return sfr_med  # Msun/yr


def ffb_sfr_sig(lgMh, z):
    return options["FFB_SFR_SIG"]  # dex


def ffb_fFFB(lgMh, z):
    """
    The meaning depends on the global variable options['FFB_FRAC_ALL']
    If true, f_ffb is defined wrt all gals, otherwise wrt SF gals.
    """
    lgMh_crit = ffb_lgMh_crit(z)
    f_FFB = sigmoid(lgMh, m=lgMh_crit, s=options["FFB_SFR_TH_SIG"], a=0, b=1)
    if options["FFB_LGMH_QUENCH"] is not None:
        f_FFB *= sigmoid(
            lgMh, m=options["FFB_LGMH_QUENCH"], s=options["FFB_SFR_TH_SIG"], a=1, b=0
        )
    return f_FFB


# UV luminosity
# -----------------------------
def func_MUV_sfr_Salpeter(SFR, z):
    kappa_UV = 1.15e-28 * options["UV_FACTOR"]
    lum = SFR / kappa_UV * (u.erg / u.s / u.Hz)
    dist_lum = 10 * u.pc
    flux = lum / (4 * pi * dist_lum**2)
    MUV = flux.to(u.ABmag).value
    return MUV


def func_MUV_sfr(SFR, z):
    # Behroozi+2020, eq B3
    kappa_UV = 5.1e-29 * (1 + exp(-20.79 / (1 + z) + 0.98))
    lum = SFR / kappa_UV * (u.erg / u.s / u.Hz)
    dist_lum = 10 * u.pc
    flux = lum / (4 * pi * dist_lum**2)
    MUV = flux.to(u.ABmag).value
    return MUV


def func_MUV_lgMs(lgMs, z):
    # Yung+2023
    MUV = -2.3 * (lgMs - 9) - 20.5 - 2.5 * log10(options["UV_FACTOR"])
    return MUV


# Combining UM and FFB
# -----------------------------
def p_lgSFR_lgMh(lgSFR, lgMh, z):
    med_ffb = ffb_sfr_med(lgMh, z)
    sig_ffb = ffb_sfr_sig(lgMh, z)
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)
    med_q = um_sfr_med_q(lgMh, z)
    sig_q = um_sfr_sig_q(lgMh, z)

    if options["FFB_FRAC_ALL"]:
        f_q_ = um_fQ(lgMh, z)
        f_ffb = ffb_fFFB(lgMh, z)
        f_um = 1 - f_ffb
        f_q = f_um * f_q_
        f_sf = f_um * (1 - f_q_)
    else:
        f_q = um_fQ(lgMh, z)
        f_FFB_ = ffb_fFFB(lgMh, z)
        f_SF_ = 1 - f_q
        f_ffb = f_SF_ * f_FFB_
        f_sf = f_SF_ * (1 - f_FFB_)

    prob = (
        f_ffb * normal(lgSFR, log10(med_ffb), sig_ffb)
        + f_sf * normal(lgSFR, log10(med_sf), sig_sf)
        + f_q * normal(lgSFR, log10(med_q), sig_q)
    )
    return prob


def func_sfr_avg(lgMh, z):
    med_ffb = ffb_sfr_med(lgMh, z)
    sig_ffb = ffb_sfr_sig(lgMh, z)
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)
    med_q = um_sfr_med_q(lgMh, z)
    sig_q = um_sfr_sig_q(lgMh, z)

    if options["FFB_FRAC_ALL"]:
        f_q_ = um_fQ(lgMh, z)
        f_ffb = ffb_fFFB(lgMh, z)
        f_um = 1 - f_ffb
        f_q = f_um * f_q_
        f_sf = f_um * (1 - f_q_)
    else:
        f_q = um_fQ(lgMh, z)
        f_FFB_ = ffb_fFFB(lgMh, z)
        f_SF_ = 1 - f_q
        f_ffb = f_SF_ * f_FFB_
        f_sf = f_SF_ * (1 - f_FFB_)

    sfr_avg = (
        f_ffb * lgnormal_mean(med_ffb, sig_ffb)
        + f_sf * lgnormal_mean(med_sf, sig_sf)
        + f_q * lgnormal_mean(med_q, sig_q)
    )
    return sfr_avg


def func_dif_sfr_avg(lgMh, z, Mdot):
    "difference of SFR between UM and FFB"
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)

    if options["FFB_FRAC_ALL"]:
        med_q = um_sfr_med_q(lgMh, z)
        sig_q = um_sfr_sig_q(lgMh, z)

        f_q = um_fQ(lgMh, z)
        f_sf = 1 - f_q
        f_ffb = ffb_fFFB(lgMh, z)

        dif_sfr_avg = f_ffb * (
            Mdot * options["FFB_SFE_MAX"]
            - f_sf * lgnormal_mean(med_sf, sig_sf)
            - f_q * lgnormal_mean(med_q, sig_q)
        )
    else:
        f_q = um_fQ(lgMh, z)
        f_FFB_ = ffb_fFFB(lgMh, z)
        f_SF_ = 1 - f_q
        f_ffb = f_SF_ * f_FFB_

        dif_sfr_avg = f_ffb * (
            Mdot * options["FFB_SFE_MAX"] - lgnormal_mean(med_sf, sig_sf)
        )
    return dif_sfr_avg.clip(0)  # min: 0


def func_lgMs_med(lgMh, z, z_max=30, n_grid=200):
    # compute Mstar from the cumulative difference btw UM and FFB
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1
    lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    Mdot_hist = 10 ** mah_der_interp(z_ob, lgM_ob, z_hist) * fb

    dif_sfr = func_dif_sfr_avg(lgM_hist, z_hist, Mdot_hist)

    dtdlgzp1_hist = (
        cosmo.lookbackTime(z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9
    )  # yr/dex

    func = Akima1DInterpolator(
        lgzp1_hist, dif_sfr * dtdlgzp1_hist, axis=1
    ).antiderivative()
    dif_Ms = func(lgzp1_hist[-1]) - func(lgzp1_hist[0])

    # UM mass and FFB correction
    Ms_um = 10 ** um_lgMs_med(lgMh, z)
    Ms = Ms_um + dif_Ms

    return log10(Ms)


def um_lgMs_med_integ(lgMh, z, z_max=30, n_grid=200):
    # compute UM Ms from cumulative SFR, for consistency check
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1

    lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    sfr = um_sfr(lgM_hist, z_hist)

    dtdlgzp1_hist = (
        cosmo.lookbackTime(z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9
    )  # yr/dex

    func = Akima1DInterpolator(lgzp1_hist, sfr * dtdlgzp1_hist, axis=1).antiderivative()
    Ms = func(lgzp1_hist[-1]) - func(lgzp1_hist[0]) + 1
    return log10(Ms)


def func_lgMh_med_integ(lgMh, z, z_max=30, n_grid=200):
    # compute halo mass from cumulative Mdot, for consistency check
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1

    # lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    Mdot = 10 ** mah_der_interp(z_ob, lgM_ob, z_hist)

    dtdlgzp1_hist = (
        cosmo.lookbackTime(z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9
    )  # yr/dex

    func = Akima1DInterpolator(
        lgzp1_hist, Mdot * dtdlgzp1_hist, axis=1
    ).antiderivative()
    Mh = func(lgzp1_hist[-1]) - func(lgzp1_hist[0]) + 1
    return log10(Mh)


def p_lgMs_lgMh(lgMs, lgMh, z):
    lgMs_med = func_lgMs_med(lgMh, z)
    lgMs_sig = um_lgMs_sig(lgMh, z)

    prob = normal(lgMs, lgMs_med, lgMs_sig)
    return prob


def func_Mdot_baryon(lgMh, z):
    if not np.isscalar(z):
        raise ValueError("z should be a scalar")

    Mdot = 10 ** mah_der_interp(z, lgMh, z) * fb
    return Mdot  # Msun/yr


def func_SFE_cumulative(lgMh, z, z_max=30):
    lgMs = func_lgMs_med(lgMh, z, z_max=z_max)
    return 10**lgMs / (10**lgMh * fb)


def func_SFE_instant(lgMh, z):
    "SFR / Mdot_baryon"
    SFR = func_sfr_avg(lgMh, z)
    Mdot_baryon = func_Mdot_baryon(lgMh, z)
    return SFR / Mdot_baryon


def p_MUV_lgMh(MUV, lgMh, z, attenuation=None):
    """
    attenuation: None, 'shell', 'disc', 'average'
    """
    assert attenuation in [None, False, "shell", "disc", "average"]
    # False or None to disable

    lgMs_med = func_lgMs_med(lgMh, z)
    lgMs_sig = um_lgMs_sig(lgMh, z)
    MUV_med = func_MUV_lgMs(lgMs_med, z=z)
    MUV_sig = ((2.3 * lgMs_sig) ** 2 + 0.3**2) ** 0.5  # 0.3? TBC

    if attenuation:
        if options["FFB_FRAC_ALL"]:
            f_ffb = ffb_fFFB(lgMh, z)
        else:
            f_FFB_ = ffb_fFFB(lgMh, z)
            f_SF_ = 1 - um_fQ(lgMh, z)
            f_ffb = f_SF_ * f_FFB_

        AUV_um = um_AUV(MUV, z=z)
        MUV_med_um = MUV_med + AUV_um

        if attenuation == "shell":
            AUV_ffb = ffb_AUV(lgMh, z, mode="shell", Zin=0.1, eps=None)
        elif attenuation == "disc":
            AUV_ffb = ffb_AUV(lgMh, z, mode="disc", Zin=0.1, eps=None)
        elif attenuation == "average":
            AUV_ffb = 0.5 * (
                ffb_AUV(lgMh, z, mode="shell", Zin=0.1, eps=None)
                + ffb_AUV(lgMh, z, mode="disc", Zin=0.1, eps=None)
            )
        MUV_med_ffb = MUV_med + AUV_ffb

        prob = f_ffb * normal(MUV, MUV_med_ffb, MUV_sig) + (1 - f_ffb) * normal(
            MUV, MUV_med_um, MUV_sig
        )
    else:
        prob = normal(MUV, MUV_med, MUV_sig)
    return prob


# Halo mass func and number densities
# -----------------------------
def compute_dNdlgMh(z, lgMh=None, return_func=False):
    "halo mass func, dN/dlogMh [cMpc^-3 dex^-1]"
    if lgMh is None:
        # nu_max = 10  # maximum peak height, lgM_max=16.7 at z=0, 11.6 at z=18
        nu_max = 8  # maximum peak height, lgM_max=16.5 at z=0, 10.8 at z=18
        Mh_max = peaks.massFromPeakHeight(nu=nu_max, z=z, **ps_ext_args) / cosmo.h
        lgMh = np.arange(5, log10(Mh_max) + 0.001, 0.025)

    dNdlnMh = mass_function.massFunction(
        10**lgMh * cosmo.h,
        z=z,
        mdef="vir",
        model="Watson13".lower(),
        q_out="dndlnM",
        **ps_ext_args,
    )  # dndlnM, comoving (Mpc/h)^-3
    dNdlgMh = dNdlnMh * cosmo.h**3 * log(10)  # comoving Mpc^-3

    if not return_func:
        return lgMh, dNdlgMh
    else:
        func_lgdNdlgMh = Akima1DInterpolator(lgMh, log10(dNdlgMh))
        return func_lgdNdlgMh


def compute_rho_dm_lgMh(z, lgMh):
    """cumulative DM density above a halo mass, rho_dm(>Mh | z)"""
    lgMh_, dNdlgMh_ = compute_dNdlgMh(z, np.linspace(8, 16.5, 101))
    F_dMh_dlgMh = Akima1DInterpolator(
        -lgMh_[::-1], (10**lgMh_ * dNdlgMh_)[::-1]
    ).antiderivative()

    rho_dm = F_dMh_dlgMh(-lgMh)
    return rho_dm


def compute_dNdlgMs(z, lgMs=None, return_func=False):
    "stellar mass func"
    lgMh, dNdlgMh = compute_dNdlgMh(z)
    if lgMs is None:
        lgMs = np.arange(5, lgMh[-1] - log10(fb) + 0.001, 0.025)

    pr_lgMs_lgMh = p_lgMs_lgMh(lgMs.reshape(-1, 1), lgMh, z)
    dNdlgMs = simps1d(pr_lgMs_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])

    if not return_func:
        return lgMs, dNdlgMs
    else:
        func_lgdNdlgMs = Akima1DInterpolator(lgMs, log10(dNdlgMs))
        return func_lgdNdlgMs


def compute_N_lgMs(z, lgMs):
    "Cumulative number of galaxies N(>lgMs | z)"
    lgMs_, dNdlgMs_ = compute_dNdlgMs(z)
    f_N_lgMs = Akima1DInterpolator(-lgMs_[::-1], dNdlgMs_[::-1]).antiderivative()
    N_lgMs = f_N_lgMs(-lgMs)
    return N_lgMs


def compute_rho_star_lgMs(z, lgMs):
    """rho_star(>Ms | z)"""
    lgMs_, dNdlgMs_ = compute_dNdlgMs(z)
    F_dMs_dlgMs = Akima1DInterpolator(
        -lgMs_[::-1], (10**lgMs_ * dNdlgMs_)[::-1]
    ).antiderivative()

    rho_star = F_dMs_dlgMs(-lgMs)
    return rho_star


def compute_rho_star(z, MUV_lim):
    """rho_star(MUV<MUV_lim | z)
    No attenuation considered yet when converting MUV_lim to Ms_lim.
    """
    lgMs, dNdlgMs = compute_dNdlgMs(z)
    F_dMs_dlgMs = Akima1DInterpolator(
        -lgMs[::-1], (10**lgMs * dNdlgMs)[::-1]
    ).antiderivative()

    MUV = func_MUV_lgMs(lgMs, z=z)
    lgMs_lim = np.interp(MUV_lim, MUV[::-1], lgMs[::-1])

    rho_star = F_dMs_dlgMs(-lgMs_lim)
    return rho_star


def compute_dNdlgSFR(z, lgSFR=None, return_func=False):
    lgMh, dNdlgMh = compute_dNdlgMh(z)
    if lgSFR is None:
        lgSFR = np.linspace(-6, 6, 121)

    pr_lgSFR_lgMh = p_lgSFR_lgMh(lgSFR.reshape(-1, 1), lgMh, z)
    dNdlgSFR = simps1d(pr_lgSFR_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])

    if not return_func:
        return lgSFR, dNdlgSFR
    else:
        func_lgdNdlgSFR = Akima1DInterpolator(lgSFR, log10(dNdlgSFR))
        return func_lgdNdlgSFR


def compute_rho_SFR(z, MUV_lim):
    "rho_SFR(MUV<MUV_lim | z)"
    lgMh, dNdlgMh = compute_dNdlgMh(z)

    lgMs = func_lgMs_med(lgMh, z)
    MUV = func_MUV_lgMs(lgMs, z)
    lgMh_lim = np.interp(MUV_lim, MUV[::-1], lgMh[::-1])

    SFR = func_sfr_avg(lgMh, z)
    F_dSFR_dlgMh = Akima1DInterpolator(
        -lgMh[::-1], (SFR * dNdlgMh)[::-1]
    ).antiderivative()

    # lgSFR = np.linspace(-6, 6, 121).reshape(-1, 1)
    # pr_lgSFR_lgMh = p_lgSFR_lgMh(lgSFR, lgMh, z)
    # dSFR_dlgMh = simps1d(
    #     10**lgSFR * pr_lgSFR_lgMh * dNdlgMh, axis=0, dx=lgSFR[1] - lgSFR[0]
    # )
    # F_dSFR_dlgMh = Akima1DInterpolator(-lgMh[::-1], dSFR_dlgMh[::-1]).antiderivative()

    rho_SFR = F_dSFR_dlgMh(-lgMh_lim)

    return rho_SFR


def compute_dNdMUV_Ms(z, MUV=None, attenuation=None, return_func=False):
    "UVLF computed using MUV-Ms relation"
    lgMh, dNdlgMh = compute_dNdlgMh(z)
    if MUV is None:
        MUV = np.arange(-28, -12 + 0.001, 0.1)

    pr_MUV_lgMh = p_MUV_lgMh(MUV.reshape(-1, 1), lgMh, z, attenuation=attenuation)
    dNdMUV = simps1d(pr_MUV_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])

    if not return_func:
        return MUV, dNdMUV
    else:
        func_lgdNdMUV = Akima1DInterpolator(MUV, log10(dNdMUV))
        return func_lgdNdMUV


def compute_rho_UV(z, MUV_lim, attenuation=None):
    "rho_UV(MUV<MUV_lim | z) in units of erg/s/Hz/Mpc^3"
    MUV, dNdMUV = compute_dNdMUV_Ms(z, attenuation=attenuation)

    dist_lum = 10 * u.pc
    flux = (MUV * u.ABmag).to(u.erg / u.s / u.cm**2 / u.Hz)
    lum = (4 * np.pi * dist_lum**2 * flux).to(u.erg / u.s / u.Hz).value
    F_dlum_dMUV = Akima1DInterpolator(MUV, lum * dNdMUV).antiderivative()

    rho_UV = F_dlum_dMUV(MUV_lim)
    return rho_UV


def convert_MUV_to_JWST(
    z,
    mag,
    band="NIRCam_F277W",
    table_path="data/Yung/Conversion_tables",
    inverse=False,
):
    """
    Tables generated by L.Y. Aaron Yung.
    Note z can only be integers within [5, 20].
    """
    from astropy.io import ascii

    IRconversion = ascii.read(
        f"{table_path}/UV_to_IR_z{z:d}.dat", names=["Conversion", "slope", "intercept"]
    )
    ix = IRconversion["Conversion"] == "UV1500_to_" + band
    IR_slope = IRconversion[ix]["slope"][0]
    IR_intercept = IRconversion[ix]["intercept"][0]

    if inverse:
        JWST_band = mag
        MUV_1500 = (JWST_band - IR_intercept) / IR_slope
        return MUV_1500
    else:
        MUV_1500 = mag
        JWST_band = MUV_1500 * IR_slope + IR_intercept
        return JWST_band


def compute_surface_density_obs(
    mag_lim,
    band="NIRCam_F277W",
    area=u.arcmin**2,
    attenuation=None,
    table_path="data/Yung/Conversion_tables",
):
    """
    N(>z, m<mlin) of bright galaxies within give area in the sky

    Returns
    -------
    z, N :
        Note z can only be integers within [5, 20].
    """
    zs = np.arange(6, 21)

    cosmo_astropy = cosmo.toAstropy()
    vol = cosmo_astropy.comoving_volume(zs).to_value(u.Mpc**3)

    mag_lim = np.atleast_1d(mag_lim)
    den_gal = np.zeros((len(zs), len(mag_lim)))  # number density at z
    num_gal = np.zeros((len(zs), len(mag_lim)))  # cumulative number > z

    for i, z in enumerate(zs):
        MUV, dNdMUV = compute_dNdMUV_Ms(z, attenuation=attenuation)
        func_den = Akima1DInterpolator(MUV, dNdMUV).antiderivative()

        MUV_lim = convert_MUV_to_JWST(
            z, mag_lim, band=band, table_path=table_path, inverse=True
        )
        den_gal[i] = func_den(MUV_lim)

    func_num = Akima1DInterpolator(-vol[::-1], den_gal[::-1]).antiderivative()
    num_gal = func_num(-vol)

    fac = (area / (4 * pi * u.steradian)).to_value(1)  # sky area
    surface_den = fac * num_gal

    return zs, surface_den


# Halo growth history
# -----------------------------
def mah_interp(z_ob, lgM_ob, z_hist):
    if options["HALO_MAH_MODEL"] == "Zhao09":
        return mah_interp_Zhao09(z_ob, lgM_ob, z_hist)
    elif options["HALO_MAH_MODEL"] == "Dekel13":
        return mah_interp_Dekel13(z_ob, lgM_ob, z_hist)
    else:
        raise ValueError("options['HALO_MAH_MODEL']' should be Dekel13 or Zhao09")


def mah_der_interp(z_ob, lgM_ob, z_hist):
    if options["HALO_MAH_MODEL"] == "Zhao09":
        return mah_der_interp_Zhao09(z_ob, lgM_ob, z_hist)
    elif options["HALO_MAH_MODEL"] == "Dekel13":
        return mah_der_interp_Dekel13(z_ob, lgM_ob, z_hist)
    else:
        raise ValueError("options['HALO_MAH_MODEL'] should be Dekel13 or Zhao09")


def mah_interp_Zhao09(z_ob, lgM_ob, z_hist):
    """
    M(z_hist|lgM_ob, z_ob)
    lgM should be between 1.5e5Msun and 8 sigma peak.
    z should be between 0 and 30.
    """
    fp = H5Attr("mandc-1.03main/run/mchistory_cDekel.h5", lazy=False)
    z_list = np.sort([fp[f"lg(z+1)/{key}/ziz"][0] for key in fp["lg(z+1)"]])
    lgzp1_list = log10(z_list + 1)

    lgzp1_ob = log10(z_ob + 1)
    lgzp1_hist = log10(z_hist + 1)

    ix_z = lgzp1_list.searchsorted(lgzp1_ob, side="right") - 1  # find closest z bin
    gp = fp[f"lg(z+1)/{lgzp1_list[ix_z]:.2f}"]

    lgM_z = log10(gp.Miz)
    lgM_i = lgM_z.T[0]
    lgzp1_i = log10(gp.ziz + 1)
    mah_interp = RegularGridInterpolator(
        [lgM_i, lgzp1_i], lgM_z, method="linear", bounds_error=False
    )

    x, y = np.broadcast_arrays(lgM_i, lgzp1_ob)
    lgM_zob = mah_interp(np.stack([x, y], axis=-1))  # lgM[z_ob | lgM_i, z_0]
    lgM_i_ = np.interp(lgM_ob, lgM_zob, lgM_i)  # lgM[z_0 | lgM_ob, z_ob]

    x, y = np.meshgrid(lgM_i_, lgzp1_hist, indexing="ij")
    lgM_hist = mah_interp(np.stack([x, y], axis=-1))

    if np.isscalar(z_hist):
        return lgM_hist.reshape(np.shape(lgM_ob))
    else:
        return lgM_hist  # shape: lgM_ob, z_hist


def mah_der_interp_Zhao09(z_ob, lgM_ob, z_hist):
    """
    dM/dt(z_hist|lgM_ob, z_ob)
    lgM should be between 1.5e5Msun and 8 sigma peak.
    z should be between 0 and 30.
    """
    fp = H5Attr("mandc-1.03main/run/mchistory_cDekel.h5", lazy=False)
    z_list = np.sort([fp[f"lg(z+1)/{key}/ziz"][0] for key in fp["lg(z+1)"]])
    lgzp1_list = log10(z_list + 1)

    lgzp1_ob = log10(z_ob + 1)
    lgzp1_hist = log10(z_hist + 1)

    ix_z = lgzp1_list.searchsorted(lgzp1_ob, side="right") - 1  # find closest z bin
    gp = fp[f"lg(z+1)/{lgzp1_list[ix_z]:.2f}"]

    lgM_z = log10(gp.Miz)
    lgMdot_z = log10(gp.Mdot)
    lgM_i = lgM_z.T[0]
    lgzp1_i = log10(gp.ziz + 1)
    mah_interp = RegularGridInterpolator(
        [lgM_i, lgzp1_i], lgM_z, method="linear", bounds_error=False
    )

    x, y = np.broadcast_arrays(lgM_i, lgzp1_ob)
    lgM_zob = mah_interp(np.stack([x, y], axis=-1))  # lgM[z_ob | lgM_i, z_0]
    lgM_i_ = np.interp(lgM_ob, lgM_zob, lgM_i)  # lgM[z_0 | lgM_ob, z_ob]

    mdot_interp = RegularGridInterpolator(
        [lgM_i, lgzp1_i], lgMdot_z, method="linear", bounds_error=False
    )

    x, y = np.meshgrid(lgM_i_, lgzp1_hist, indexing="ij")
    lgMdot_hist = mdot_interp(np.stack([x, y], axis=-1))

    if np.isscalar(z_hist):
        return lgMdot_hist.reshape(np.shape(lgM_ob))
    else:
        return lgMdot_hist  # shape: lgM_ob, z_hist


def mah_interp_Dekel13(z_ob, lgM_ob, z_hist):
    s = 0.03  # 0.03 Gyr^-1
    t1 = 17.00  # Gyr, H0=70, omegam=0.3
    alpha = 1.5 * s * t1

    lgM_ob_T = (
        lgM_ob if np.isscalar(lgM_ob) else lgM_ob.reshape(-1, 1)
    )  # transpose of lgM_ob

    if options["DEKEL13_BETA"] == 0:
        lgM_hist = lgM_ob_T - alpha * (z_hist - z_ob) / log(10)
    else:
        beta, lgM_c0 = options["DEKEL13_BETA"], abs(options["DEKEL13_PIVOT"])
        if options["DEKEL13_PIVOT"] > 0:
            base = 10 ** (-beta * (lgM_ob_T - lgM_c0)) + alpha * beta * (z_hist - z_ob)
            lgM_hist = lgM_c0 - log10(base) / beta
        else:
            # negative for varying pivot, for my own test only
            base = (
                10 ** (-beta * (lgM_ob_T - lgM_c0))
                + exp(alpha * beta * (z_hist - z_ob))
                - 1
            )
            lgM_hist = lgM_c0 - log10(base) / beta

    if np.isscalar(z_hist):
        return lgM_hist.reshape(np.shape(lgM_ob))
    else:
        return lgM_hist  # shape: lgM_ob, z_hist


def mah_der_interp_Dekel13(z_ob, lgM_ob, z_hist):
    s = 0.03  # 0.03 Gyr^-1
    t1 = 17.00  # Gyr, H0=70, omegam=0.3
    alpha = 1.5 * s * t1

    lgM_hist = mah_interp_Dekel13(z_ob, lgM_ob, z_hist)
    lgMdot_hist = lgM_hist + log10(s * (1 + z_hist) ** 2.5) - 9

    if options["DEKEL13_BETA"] != 0:
        beta, lgM_c0 = options["DEKEL13_BETA"], abs(options["DEKEL13_PIVOT"])
        if options["DEKEL13_PIVOT"] > 0:
            lgM_c = lgM_c0
        else:
            lgM_c = lgM_c0 - alpha * (z_hist - z_ob) / log(10)
        lgMdot_hist += beta * (lgM_hist - lgM_c)

    if np.isscalar(z_hist):
        return lgMdot_hist.reshape(np.shape(lgM_ob))
    else:
        return lgMdot_hist  # shape: lgM_ob, z_hist


# FFB steady wind -- cooling
# -----------------------------
def ffb_wind_speed(lgMh=None, z=None, eps=None):
    "either provide (lgMh, z) or eps"
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    eta = 5 / eps - 4
    Vwind = 3333 * eta**-0.5  # km/s
    return Vwind


def ffb_rcool(lgMh, z, mode="shell", eps=None):
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    eta = 5 / eps - 4
    rshell = ffb_radius(lgMh, z, mode=mode, lambdas=0.025, eps=eps)
    # SFR = func_sfr_avg(lgMh, z)
    SFR = func_Mdot_baryon(lgMh, z) * eps
    rcool = (
        4 * (0.2 * eta) ** -2.92 * (rshell / 0.3) ** 1.79 * (0.3 * SFR / 10) ** -0.79
    )
    return rcool


# FFB galaxy size
# -----------------------------
def ffb_radius(lgMh, z, mode="shell", lambdas=0.025, eps=None):
    """
    Galaxy radius.
    Important note: this radius is about 2Re!
    """
    if mode == "shell":
        if eps is None:
            eps = func_SFE_instant(lgMh, z)
        eta = 5 / eps - 4

        Mz_dep = 10 ** ((lgMh - 10.8) * (1 / 6)) * ((1 + z) / 10) ** -0.75
        radius = 0.56 * (lambdas / 0.025) * (eta**0.25 * eps**0.5) * Mz_dep
    elif mode == "disc":
        Mz_dep = 10 ** ((lgMh - 10.8) / 3) * ((1 + z) / 10) ** -1
        radius = 0.62 * (lambdas / 0.025) * Mz_dep
    return radius


# FFB gas fraction
# -----------------------------
def ffb_Mgas(lgMh, z, mode="shell", eps=None):
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    eta = 5 / eps - 4

    R = ffb_radius(lgMh, z, mode=mode, lambdas=0.025, eps=eps)
    SFR = func_Mdot_baryon(lgMh, z) * eps
    # SFR = func_sfr_avg(lgMh, z) # equivalent

    mgas = 1.04e5 * eta**1.5 * SFR * R
    return mgas.clip(0, 10**lgMh)


def ffb_fgas(lgMh, z, mode="shell", eps=None):
    "gas fraction: mgas / (mgas + mstar)"
    mgas = ffb_Mgas(lgMh, z, mode=mode, eps=eps)
    if eps is None:
        mstar = 10 ** func_lgMs_med(lgMh, z)
    else:
        mstar = fb * 10**lgMh * eps
    return mgas / (mgas + mstar)


# The four functions below are approximations, which are not used in practice
def ffb_ratio_Mgas_star_shell(lgMh, z, eps=1):
    "approximate gas ratio"
    eta = 5 / eps - 4
    Mz_dep = 10 ** ((lgMh - 10.8) * 0.31) * ((1 + z) / 10) ** 1.75
    return 0.61e-2 * (eta / 6) ** 1.75 * (eps / 0.5) ** 0.5 * Mz_dep


def ffb_ratio_Mgas_star_disk(lgMh, z, eps=1):
    "approximate gas ratio"
    eta = 5 / eps - 4
    Mz_dep = 10 ** ((lgMh - 10.8) * 0.47) * ((1 + z) / 10) ** 1.5
    return 0.61e-2 * (eta / 6) ** 1.5 * Mz_dep


def ffb_ratio_Mgas_star_shell_Mcrit(z, eps=1):
    "approximate gas ratio"
    eta = 5 / eps - 4
    Mz_dep = ((1 + z) / 10) ** -0.15
    return 0.61e-2 * (eta / 6) ** 1.75 * (eps / 0.5) ** 0.5 * Mz_dep


def ffb_ratio_Mgas_star_disk_Mcrit(z, eps=1):
    "approximate gas ratio"
    eta = 5 / eps - 4
    Mz_dep = ((1 + z) / 10) ** -1.43
    return 0.61e-2 * (eta / 6) ** 1.5 * Mz_dep


# FFB metallicity
# -----------------------------
def ffb_str_coverage(lgMh, z, eps=None):
    "Sky coverage fraction by streams"
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    eta = 5 / eps - 4

    Mz_dep = 10 ** ((lgMh - 10.8) * 0.3333) * ((1 + z) / 10) ** 0.5
    f_omega = 0.22 * eta**-0.5 * eps**-1 * Mz_dep
    return f_omega.clip(0, 1)


def ffb_metal(lgMh, z, Zsn=1, Zin=0.1, eps=None):
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    f_omega = ffb_str_coverage(lgMh, z, eps=eps)

    Zmix = Zin + 0.2 * eps * f_omega / (1 + (1 - 0.8 * eps) * f_omega) * (Zsn - Zin)
    return Zmix


# FFB dust attenuation
# -----------------------------
def AUV_to_tau(tau):
    "convert tau to AUV"
    return 2.5 * tau / log(10)


def fobsc_to_AUV(fobsc):
    return -2.5 * np.log10(1 - fobsc)


def AUV_to_fobsc(AUV):
    return 1 - 10 ** (AUV / -2.5)


def ffb_f_sfe(sfe):
    return (5 * sfe * (1 - 0.8 * sfe)) ** 0.5


def ffb_tau(lgMh, z, mode="shell", Zin=0, eps=None):
    if eps is None:
        eps = func_SFE_instant(lgMh, z)
    eta = 5 / eps - 4

    R = ffb_radius(lgMh, z, mode=mode, lambdas=0.025, eps=eps)
    SFR = func_Mdot_baryon(lgMh, z) * eps
    # SFR = func_sfr_avg(lgMh, z) # equivalent

    f_dsn = 6.5
    if Zin == 0:
        f_d = f_dsn
    else:
        f_d = f_dsn + 5 * (1 / eps - 1) * Zin**1.6

    if mode == "shell":
        neg_log_meanexp = lambda x, y: -log(0.5 * (exp(-x) + exp(-y)))
        fac = neg_log_meanexp(3.08 + 0.52, 0.52)  # 0.5(>0, >R)
        # fac = -log(0.5 * (exp(-3.08 - 0.52) + exp(-0.52)))
    elif mode == "disc":
        fac = 0.52  # (>R)

    tau = fac * 1e-3 * f_d * eta**0.5 / R * SFR
    return tau


def ffb_AUV(lgMh, z, mode="shell", Zin=0, eps=None):
    tau = ffb_tau(lgMh, z, mode=mode, Zin=Zin, eps=eps)
    AUV = AUV_to_tau(tau)
    return AUV


def ffb_fobsc(lgMh, z, mode="shell", Zin=0, eps=None):
    tau = ffb_tau(lgMh, z, mode=mode, Zin=Zin, eps=eps)
    return 1 - np.exp(-tau)


# The four functions below are approximations, which are not used in practice
def ffb_tau_shell(sfe, lgMh, z):
    "UV optical depth for shell scenario, Li+23, eq 48"
    # for the shell the average of r>0 and r>R
    Mz_dep = 10 ** (0.97 * (lgMh - 10.8)) * ((1 + z) / 10) ** 3.25
    neg_log_meanexp = lambda x, y: -log(0.5 * (exp(-x) + exp(-y)))
    tau = neg_log_meanexp(2.32 + 0.39, 0.39) * ffb_f_sfe(sfe) ** 0.5 * Mz_dep
    return tau


def ffb_tau_disc(sfe, lgMh, z):
    "UV optical depth for disc scenario, Li+23, eq 48"
    # for the disc only r>R
    Mz_dep = 10 ** (0.81 * (lgMh - 10.8)) * ((1 + z) / 10) ** 3.5
    tau = (0.36) * ffb_f_sfe(sfe) * Mz_dep
    return tau


def ffb_AUV_shell(lgMh, z):
    sfe = func_SFE_instant(lgMh, z)
    tau = ffb_tau_shell(sfe, lgMh, z)
    AUV = AUV_to_tau(tau)
    return AUV


def ffb_AUV_disc(lgMh, z):
    sfe = func_SFE_instant(lgMh, z)
    tau = ffb_tau_disc(sfe, lgMh, z)
    AUV = AUV_to_tau(tau)
    return AUV


# integration functions
# -----------------------------
# author: Zhaozhou Li
# source: https://github.com/syrte/handy/blob/master/integrate.py


def slice_set(ix, ndim, axis):
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    ix_list = [slice(None)] * ndim
    ix_list[axis] = ix
    return tuple(ix_list)


def sum2d(a):
    """sum of last two dimensions"""
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    return a.reshape(*a.shape[:-2], -1).sum(-1)


def simps1d(y, dx=1.0, axis=-1, even="avg"):
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    y = np.asarray(y)
    ndim = y.ndim

    # when shape of y is odd
    if y.shape[axis] % 2 == 1:
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(-1, ndim, axis)
        ixo = slice_set(slice(1, -1, 2), ndim, axis)  # odd
        ixe = slice_set(slice(2, -2, 2), ndim, axis)  # even
        out = (y[ix0] + y[ix1] + 4 * y[ixo].sum(axis) + 2 * y[ixe].sum(axis)) * (dx / 3)
        return out
    elif even == "avg":
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(-1, ndim, axis)
        ix2 = slice_set(1, ndim, axis)
        ix3 = slice_set(-2, ndim, axis)
        ix4 = slice_set(slice(2, -2), ndim, axis)
        out = (
            2.5 * (y[ix0] + y[ix1]) + 6.5 * (y[ix2] + y[ix3]) + 6 * y[ix4].sum(axis)
        ) * (dx / 6)
        return out
    elif even == "first":
        ix0 = slice_set(-1, ndim, axis)
        ix1 = slice_set(-2, ndim, axis)
        ix3 = slice_set(slice(None, -1), ndim, axis)
        return simps1d(y[ix3], dx, axis) + 0.5 * dx * (y[ix0] + y[ix1])
    elif even == "last":
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(1, ndim, axis)
        ix3 = slice_set(slice(1, None), ndim, axis)
        return simps1d(y[ix3], dx, axis) + 0.5 * dx * (y[ix0] + y[ix1])
    else:
        raise ValueError("'even' must be one of 'avg', 'first' or 'last'")


def simps2d(z, dx=1, dy=1):
    """integrate over last two dimensions

    >>> simps2d(np.ones((5, 5)))
    16.0
    """
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    z = np.asarray(z)
    nx, ny = z.shape[-2:]
    if nx % 2 != 1 or ny % 2 != 1:
        raise ValueError("input array should be odd shape")

    ixo = slice(1, -1, 2)  # odd
    ixe = slice(2, -2, 2)  # even

    # corner points, with weight 1
    s1 = z[..., 0, 0] + z[..., 0, -1] + z[..., -1, 0] + z[..., -1, -1]

    # edges excluding corners, with weight 2 or 4
    s2 = 2 * (
        z[..., 0, ixe].sum(-1)
        + z[..., -1, ixe].sum(-1)
        + z[..., ixe, 0].sum(-1)
        + z[..., ixe, -1].sum(-1)
    )
    s3 = 4 * (
        z[..., 0, ixo].sum(-1)
        + z[..., -1, ixo].sum(-1)
        + z[..., ixo, 0].sum(-1)
        + z[..., ixo, -1].sum(-1)
    )

    # interior points, with weight 4, 8 or 16
    s4 = (
        4 * sum2d(z[..., ixe, ixe])
        + 16 * sum2d(z[..., ixo, ixo])
        + 8 * sum2d(z[..., ixe, ixo])
        + 8 * sum2d(z[..., ixo, ixe])
    )

    out = (s1 + s2 + s3 + s4) * (dx * dy / 9)
    return out


def amap(func, *args):
    """Array version of build-in map
    amap(function, sequence[, sequence, ...]) -> array
    Examples
    --------
    >>> amap(lambda x: x**2, 1)
    array(1)
    >>> amap(lambda x: x**2, [1, 2])
    array([1, 4])
    >>> amap(lambda x,y: y**2 + x**2, 1, [1, 2])
    array([2, 5])
    >>> amap(lambda x: (x, x), 1)
    array([1, 1])
    >>> amap(lambda x,y: [x**2, y**2], [1,2], [3,4])
    array([[1, 9], [4, 16]])
    """
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/misc.py
    args = np.broadcast(*args)
    res = np.array([func(*arg) for arg in args])
    shape = args.shape + res.shape[1:]
    if shape == ():
        return res[0]
    else:
        return res.reshape(shape)


# obsolete functions
# -----------------------------
def _func_dN_dlgMsub_cond(lgMsub, lgMh):
    "unevolved subhalo mass funct, Han+ 2017, HBT+, table 1 and eq 4"
    a1, alpha1, a2, alpha2, b, beta = 0.11, 0.95, 0.32, 0.08, 8.9, 1.9
    mu = 10 ** (lgMsub - lgMh)
    dndlgmu = (a1 * mu**-alpha1 + a2 * mu**-alpha2) * exp(-b * mu**beta) * log(10)
    return dndlgmu


class _HaloMassFunc:
    def __init__(self, z, compute_sub=False):
        nu_max = 10  # maximum peak height, lgM_max=16.7 at z=0, 11.6 at z=18
        nu_max = 8  # maximum peak height, lgM_max=16.5 at z=0, 10.8 at z=18
        self.lgMh_max = log10(
            peaks.massFromPeakHeight(nu=nu_max, z=z, **ps_ext_args) / cosmo.h
        )
        self.z = z

        # central MF
        lgMcen = np.arange(5, self.lgMh_max + 0.001, 0.025)
        dNdlgMcen = (
            mass_function.massFunction(
                10**lgMcen * cosmo.h,
                z=z,
                mdef="vir",
                model="Watson13".lower(),
                q_out="dndlnM",
                **ps_ext_args,
            )
            * cosmo.h**3
            * log(10)
        )  # comoving Mpc^-3
        self.func_lgdNdlgM_cen = Akima1DInterpolator(lgMcen, log10(dNdlgMcen))
        self.lgMcen, self.dNdlgMcen = lgMcen, dNdlgMcen

        # not used
        if compute_sub:
            # prepare grids
            x0, w0 = roots_legendre(80)
            x0, w0 = x0 * 0.5 + 0.5, w0 * 0.5

            # sub MF
            lgMsub = lgMcen
            lgMcen_ = lgMsub + (self.lgMh_max - lgMsub) * x0.reshape(-1, 1)
            dNdlgMsub_cond = _func_dN_dlgMsub_cond(lgMsub, lgMcen_)
            dNdlgMcen_ = 10 ** self.func_lgdNdlgM_cen(lgMcen_)
            dNdlgMsub = (dNdlgMsub_cond * dNdlgMcen_ * w0.reshape(-1, 1)).sum(0)
            self.func_lgdNdlgM_sub = Akima1DInterpolator(lgMsub, log10(dNdlgMsub))

    def compute_dNdlgSFR(self, lgSFR=None):
        if lgSFR is None:
            lgSFR = np.linspace(-6, 6, 121)
        lgMh, dNdlgMh = self.lgMcen, self.dNdlgMcen
        pr_lgSFR_lgMh = p_lgSFR_lgMh(lgSFR.reshape(-1, 1), lgMh, self.z)
        dNdlgSFR = simps1d(pr_lgSFR_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])
        # self.func_lgdNdlgSFR = Akima1DInterpolator(lgSFR, log10(dNdlgSFR))
        # self.lgSFR, self.dNdlgSFR = lgSFR, dNdlgSFR
        return lgSFR, dNdlgSFR

    def compute_dNdlgMs(self, lgMs=None):
        lgMh, dNdlgMh = self.lgMcen, self.dNdlgMcen
        if lgMs is None:
            lgMs = np.arange(5, self.lgMh_max - log10(fb) + 0.001, 0.025)
        pr_lgMs_lgMh = p_lgMs_lgMh(lgMs.reshape(-1, 1), lgMh, self.z)
        dNdlgMs = simps1d(pr_lgMs_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])
        # self.func_lgdNdlgMs = Akima1DInterpolator(lgMs, log10(dNdlgMs))
        # self.lgMs, self.dNdlgMs = lgMs, dNdlgMs
        return lgMs, dNdlgMs

    def compute_dNdMUV_Ms(self, MUV=None, attenuation=None):
        lgMh, dNdlgMh = self.lgMcen, self.dNdlgMcen
        if MUV is None:
            MUV = np.arange(-28, -15 + 0.001, 0.1)
        pr_MUV_lgMh = p_MUV_lgMh(
            MUV.reshape(-1, 1), lgMh, self.z, attenuation=attenuation
        )
        dNdMUV = simps1d(pr_MUV_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])
        # self.func_lgdNdMUV = Akima1DInterpolator(MUV, log10(dNdMUV))
        # self.MUV, self.dNdMUV = MUV, dNdMUV
        return MUV, dNdMUV

    def compute_N_lgMs(self, lgMs):
        lgMs_, dNdlgMs_ = self.compute_dNdlgMs()
        f_N_lgMs = Akima1DInterpolator(-lgMs_[::-1], dNdlgMs_[::-1]).antiderivative()
        N_lgMs = f_N_lgMs(-lgMs)
        return N_lgMs

    def compute_dNdMUV_SFR_old(self):
        # obsolete
        MUV = func_MUV_sfr(10**self.lgSFR, z=self.z)
        dNdMUV = self.dNdlgSFR / 2.5
        self.func_lgdNdMUV_SFR = Akima1DInterpolator(MUV[::-1], log10(dNdMUV)[::-1])
        self.MUV_SFR, self.dNdMUV_SFR = MUV, dNdMUV

        # dust extinction
        AUV, dAUV_dMUV = um_AUV(MUV, z=self.z, derivative=True)
        self.MUV_SFR_obs = MUV + AUV
        self.dNdMUV_SFR_obs = dNdMUV / np.abs(1 + dAUV_dMUV)
        self.func_lgdNdMUV_SFR_obs = Akima1DInterpolator(
            self.MUV_SFR_obs[::-1], log10(self.dNdMUV_SFR_obs)[::-1]
        )

    def compute_dNdMUV_Ms_old(self):
        # obsolete
        MUV = func_MUV_lgMs(self.lgMs, z=self.z)
        dNdMUV = self.dNdlgMs / 2.3  # XXX
        self.func_lgdNdMUV_Ms = Akima1DInterpolator(MUV[::-1], log10(dNdMUV)[::-1])
        self.MUV_Ms, self.dNdMUV_Ms = MUV, dNdMUV

        # dust extinction
        AUV, dAUV_dMUV = um_AUV(MUV, z=self.z, derivative=True)
        self.MUV_Ms_obs = MUV + AUV
        self.dNdMUV_Ms_obs = dNdMUV / np.abs(1 + dAUV_dMUV)
        self.func_lgdNdMUV_Ms_obs = Akima1DInterpolator(
            self.MUV_Ms_obs[::-1], log10(self.dNdMUV_Ms_obs)[::-1]
        )


def _ffb_rdisc(lgMh, z):
    """
    Mh: Msun
    z:
    rdisc: kpc
    """
    rdisc = 0.31 * 10 ** ((lgMh - 10.8) / 3) * ((1 + z) / 10) ** -1
    return rdisc


def _ffb_rshell2(lgMh, z):
    """
    Mh: Msun
    z:
    rshell: kpc
    Note Re is about half of rshell
    obsolete estimate
    """
    rshell = 0.79 * 10 ** ((lgMh - 10.8) * -0.06) * ((1 + z) / 10) ** -2.5
    return rshell


def _ffb_rshell(lgMh, z, eps=1):
    """
    Mh: Msun
    z:
    rshell: kpc
    Note Re is about half of rshell
    """
    eta = 5 / eps - 4
    rshell = (
        0.56
        * eta**0.25
        * eps**0.5
        * 10 ** ((lgMh - 10.8) * (1 / 6))
        * ((1 + z) / 10) ** -0.75
    )
    return rshell


def _ffb_rdisk_Mcrit(z):
    return 0.29 * ((1 + z) / 10) ** -3.07


def _ffb_rshell2_Mcrit(z):
    return 0.79 * ((1 + z) / 10) ** -2.13


def _ffb_rshell_Mcrit(z, eps=1):
    eta = 5 / eps - 4
    return 0.56 * eta**0.25 * eps**0.5 * ((1 + z) / 10) ** -1.78


def _ffb_lgMcrit_disc(z):
    "Halo mass threshold for FFB"
    lgMh_crit = 10.8 + log10(0.8) - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit


def _ffb_lgMcrit_shell(z):
    "Halo mass threshold for FFB"
    lgMh_crit = 10.8 + log10(1) - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit

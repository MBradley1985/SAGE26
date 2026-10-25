"""Shared loading and the SAGE26 SFprescription = 1 recipe re-implemented in numpy.

The recipe mirrors src/model_h2_chemistry.c (BR06 pressure law, radial
integration, 25 bins out to 5 r_s) and src/model_starformation_and_feedback.c
(SFR = SfrEfficiency * M_H2 / t_dyn, t_dyn = 3 r_s / V_vir).
"""
import numpy as np

SP = '/private/tmp/claude-503/-Users-mbradley-Documents-PhD-SAGE26/ae0e8982-6ecb-41bb-84b1-f1c3f60a057d/scratchpad/'
H = 0.6774
KM_PER_PC = 3.0857e13
SEC_PER_YR = 3.15576e7
X_H = 0.74                      # hydrogen mass fraction used by SAGE26 (HYDROGEN_MASS_FRAC)


def load(snap='Snap_48'):
    d = dict(np.load(SP + f'micro_{snap}.npz'))
    m = lambda k: d[k].astype('f8') * 1e10 / H
    g = dict(
        ms=m('StellarMass'), mb=m('BulgeMass'), mcold=m('ColdGas'), mh2=m('H2gas'), mhi=m('H1gas'),
        zcold=m('MetalsColdGas'), zstar=m('MetalsStellarMass'),
        sfr=(d['SfrDisk'] + d['SfrBulge']).astype('f8'), sfr_disk=d['SfrDisk'].astype('f8'),
        rs_pc=d['DiskRadius'].astype('f8') * 1e6 / H, vvir=d['Vvir'].astype('f8'),
        rvir_pc=d['Rvir'].astype('f8') * 1e6 / H, mvir=m('Mvir'), type=d['Type'],
        loading=d['MassLoading'].astype('f8'),
        mhot=m('HotGas'), zhot=m('MetalsHotGas'), mcgm=m('CGMgas'), zcgm=m('MetalsCGMgas'),
        mej=m('EjectedMass'), zej=m('MetalsEjectedMass'), mics=m('IntraClusterStars'), zics=m('MetalsIntraClusterStars'),
        gid=d['GalaxyIndex'], cid=d['CentralGalaxyIndex'])
    g['ssfr'] = g['sfr'] / np.maximum(g['ms'], 1.0)
    g['sf'] = g['ssfr'] > 1e-11
    g['logms'] = np.log10(np.maximum(g['ms'], 1.0))
    return g


def h2_br06(mcold, mdisk_star, rs_pc, P0=4.3e4, alpha=0.92, chi=1.0, nbins=25, rmax=5.0, vg=8.0):
    """BR06 H2 mass [Msun] of an exponential disk, as SAGE26 integrates it.

    chi scales the gas scale length relative to the stellar one (chi = 1 in SAGE26).
    """
    rs_g = chi * rs_pc
    sg0 = mcold / (2 * np.pi * rs_g ** 2)
    ss0 = np.where(mdisk_star > 0, mdisk_star / (2 * np.pi * rs_pc ** 2), 0.0)
    hstar = 10 ** (-0.23 + 0.8 * np.log10(rs_pc))
    rmax_pc = rmax * rs_g
    dr = rmax_pc / nbins
    out = np.zeros_like(mcold)
    for i in range(nbins):
        r = (i + 0.5) * dr
        sg = sg0 * np.exp(-r / rs_g)
        ss = np.maximum(ss0 * np.exp(-r / rs_pc), 0.1)
        p = 272.0 * sg * np.sqrt(ss) * vg / np.sqrt(hstar)
        rmol = (p / P0) ** alpha
        fmol = rmol / (1 + rmol)
        out += np.where(sg >= 1e-3, 2 * np.pi * r * sg * X_H * fmol * dr, 0.0)
    return out


def tdyn_yr(rs_pc, vvir):
    return 3.0 * rs_pc * KM_PER_PC / vvir / SEC_PER_YR


def sfr_model(mcold, mdisk_star, rs_pc, vvir, eff=0.05, **kw):
    return eff * h2_br06(mcold, mdisk_star, rs_pc, **kw) / tdyn_yr(rs_pc, vvir)


def solve_gas(target_sfr, sfr_of_gas, lo, hi, n=60):
    """Bisect (in log) for the cold-gas mass whose SFR equals target_sfr. Vectorised."""
    a, b = np.log(lo), np.log(hi)
    for _ in range(n):
        mid = 0.5 * (a + b)
        s = sfr_of_gas(np.exp(mid))
        too_much = s > target_sfr
        b = np.where(too_much, mid, b)
        a = np.where(too_much, a, mid)
    return np.exp(0.5 * (a + b))


def binned(x, y, edges, sel=None, stat=np.median, nmin=20):
    if sel is None:
        sel = np.ones_like(x, bool)
    c = 0.5 * (edges[1:] + edges[:-1])
    out = np.full((len(c), 3), np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = sel & (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() >= nmin:
            out[i] = [stat(y[m]), np.percentile(y[m], 16), np.percentile(y[m], 84)]
    return c, out

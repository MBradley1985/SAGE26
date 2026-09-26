#!/usr/bin/env python
"""
Cold-gas number density of high-redshift SAGE26 galaxies, against the
Dekel et al. (2023) feedback-free-burst density threshold.

Dekel+23 (their Section 2) show that a starburst proceeds free of stellar-wind
and supernova feedback whenever the free-fall time is shorter than the ~1 Myr
delay before that feedback becomes effective (eq. 3), which -- via their
eq. (4) free-fall time / density relation -- is equivalent to a gas density

    n > n_fbk = 2.23e3 cm^-3                                        (eq. 5)

This script computes the mean cold-gas number density of each galaxy,
n = ColdGas / [pi Rd^2 H_disk] / (mu m_p) with mu = 1.2 (their adopted mean
molecular weight for neutral H+He gas at T < 10^4 K) and Rd SAGE26's own
exponential scale radius DiskRadius, used directly (no conversion to
Dekel+23's Re effective-radius convention in eq. 37), and plots it against
stellar mass for every output snapshot in [--zmin, --zmax] (coloured
continuously by redshift), with n_fbk drawn as a horizontal line. This is
the density that FeedbackFreeModeOn=8 (model_regimes.c) effectively compares
against once boosted by a clumping factor -- FFBCloudClumping -- from the
galaxy's mean disc density up to the density of its actual star-forming
clumps; this plot shows where the raw, unclumped disc density itself already
sits relative to n_fbk.

The disc is modelled as a cylinder, not a sphere: SAGE26 has no independently
computed gas-disc scale height, so H_disk = DISC_HEIGHT_FACTOR * Rd with
DISC_HEIGHT_FACTOR = 1/3 (thin-disc assumption -- see that constant's comment
below for what SAGE26 does and does not already compute here). This is
distinct from the *stellar* scale height calculate_stellar_scale_height_BR06()
computes in model_h2_chemistry.c (Blitz & Rosolowsky 2006 eq. 9,
h* = 10^(-0.23 - 0.8 log10 R*) pc) for the H2 midplane-pressure recipe -- a
different quantity, fit to the stellar disc for a different purpose, not a
gas-disc scale height available to reuse here.

A second figure scatters every galaxy on the (redshift, halo mass)
plane, coloured by that same cold-gas density, against Dekel+23's actual FFB
threshold curve in this plane (their eq. 62, Sec. 8): the mass above which a
halo is expected to host an FFB, M_v,ffb(z) = 10^10.8 Msun ((1+z)/10)^-6.2 --
the same functional form (and default exponent) as calculate_ffb_threshold_mass()
in model_regimes.c. The individual galaxies whose own n crosses n_fbk (eq. 5)
are ringed in red -- these need not lie above the eq. 62 curve, since that
curve is the mass threshold for the density-averaged *shell/disc* scenario,
not a restatement of this plot's own per-galaxy ColdGas/DiskRadius density.

Both figures can optionally exclude a low-spin outlier tail before plotting
(off by default; see MIN_SPIN_PERCENTILE / --min-spin-percentile):
get_disk_radius() in model_misc.c sets DiskScaleRadius proportional to the
halo's own spin parameter with no floor, so the rare halo that lands in the
low-spin tail of the (roughly mass-independent) spin distribution gets an
arbitrarily small disc and, via n ~ M/R^3, an arbitrarily large density --
independent of whether it is remotely massive enough to actually host an
FFB. Because the halo mass function is steep, this tail is numerically
dominated by low-mass haloes even though the underlying mechanism has
nothing to do with mass, which can swamp the (real, and increasing-with-mass)
bulk n-Mvir relation if not excluded.

    python plotting/ffb_cold_gas_density.py
    python plotting/ffb_cold_gas_density.py --dir output/millennium_noffb --zmin 6 --zmax 15
    python plotting/ffb_cold_gas_density.py --min-mstar 0   # include every galaxy, not just m* > 1e8
    python plotting/ffb_cold_gas_density.py --min-spin-percentile 1   # drop the bottom 1% by spin
    python plotting/ffb_cold_gas_density.py --min-particles 32   # drop poorly-resolved haloes
    python plotting/ffb_cold_gas_density.py --evolving-clumping --clumping-evolve-with fire
    python plotting/ffb_cold_gas_density.py --grid   # FFBClumpingLawGrid.pdf, all clumping laws
    python plotting/ffb_cold_gas_density.py --shell  # every figure from D23's shell density

SAGE26 -- released under MIT (see LICENSE).
"""

import argparse, glob, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'kieren_cohare_palatino_sty.mplstyle')

MSUN_G = 1.989e33          # solar mass [g]
PROTON_MASS_G = 1.6726e-24 # proton mass [g]
MU_NEUTRAL = 1.2           # Dekel+23 sec. 2: mean molecular weight, neutral H+He, T < 10^4 K

# Dekel et al. (2023) eq. (5): feedback-free density threshold.
N_FBK_CM3 = 2.23e3

# Dekel et al. (2023) eq. (62) (Sec. 8, "shell" scenario; the disc scenario's
# eq. 67 gives an equivalent curve): the FFB threshold in the halo-mass/
# redshift plane, M_v,10.8 * ((1+z)/10)^6.2 > 1, i.e.
#     M_v,ffb(z) = 10^10.8 Msun * ((1+z)/10)^-6.2.
# Combines their eq. (5) density threshold with the post-shock shell density
# scaling of eq. (41) -- an explicit M-z curve, unlike n_fbk itself, which is
# a single number with no mass or redshift dependence baked in. Same
# functional form (and default exponent) as calculate_ffb_threshold_mass()
# in model_regimes.c, credited there to Li et al. (2024) eq. 2.
FFB_THRESHOLD_NORM_LOG_MSUN = 10.8
FFB_THRESHOLD_SLOPE = -6.2


def ffb_threshold_mass_msun(z):
    """Dekel+23 eq. (62): FFB threshold halo mass [Msun] at redshift z."""
    return 10.0 ** (FFB_THRESHOLD_NORM_LOG_MSUN
                     + FFB_THRESHOLD_SLOPE * np.log10((1.0 + z) / 10.0))

# Width (in dex of Mvir) of the smooth sigmoid straddling M_v,ffb(z), Li et
# al. (2024) eq. (3) -- exactly what calculate_ffb_fraction() in
# model_regimes.c uses for FeedbackFreeModeOn in {1, 6}. A halo-mass-only
# criterion, with no dependence on any simulated gas structure.
FFB_SIGMOID_DELTA_DEX = 0.15


def ffb_sigmoid_fraction(logmv, z, delta_dex=FFB_SIGMOID_DELTA_DEX):
    """Li+2024 eq. (3): sigmoid FFB probability at (Mvir, z), matching
    calculate_ffb_fraction() in model_regimes.c exactly -- 1/(1+exp(-x)) with
    x = log10(Mvir / M_v,ffb(z)) / delta_dex, so the transition is centred on
    the eq. 62/eq. 2 threshold curve and half-width delta_dex in log-mass.
    """
    log_mthresh = np.log10(ffb_threshold_mass_msun(z))
    x = (logmv - log_mthresh) / delta_dex
    return 1.0 / (1.0 + np.exp(-x))

# Thin-disc scale height as a fraction of DiskRadius, SAGE26's exponential
# scale radius. SAGE26 does not compute a gas-disc scale height anywhere --
# DiskRadius is a 2D exponential scale length, with no vertical structure
# attached to it. The only scale height in the codebase is
# calculate_stellar_scale_height_BR06() in model_h2_chemistry.c, an empirical
# *stellar* scale height (Blitz & Rosolowsky 2006 eq. 9) used solely by the
# BR06 H2 midplane-pressure recipe, not a gas quantity available to reuse
# here. 1/3 is a generic thin-disc assumption (e.g. Kregel, van der Kruit &
# de Grijs 2002 find R_d/H_d ~ 3-5 for local spirals), not a value calibrated
# against SAGE26 itself.
DISC_HEIGHT_FACTOR = 1.0 / 3.0

# SAGE26's own actual FFB density criterion (FeedbackFreeModeOn=8,
# model_regimes.c) evaluates n within Re = EFFECTIVE_RADIUS_FACTOR *
# DiskScaleRadius, not raw DiskRadius (this script's re_factor=1.0 default) --
# matching EFFECTIVE_RADIUS_FACTOR in model_regimes.c exactly. Used as the
# fixed "radius correction" for every panel of figure_clumping_law_grid().
EFFECTIVE_RADIUS_FACTOR = 1.68

PC_TO_CM = 3.0857e18       # parsec [cm]
KM_TO_CM = 1.0e5
GYR_TO_S = 3.1557e16

# --- Dekel+23 shell scenario (their sec. 8.1, eqs. 38-41) ---------------
# The eq. 62 threshold this script draws is NOT built on the disc density at
# all: it is built on the POST-SHOCK SHELL density (eq. 41), the density of
# cosmologically accreting gas after it is shocked at the stream/disc
# interface. D23 sec. 8.2 is explicit that the two differ -- "the basic
# condition for FFB in a disc, n_d > n_fbk, translates to a threshold in
# redshift with no explicit mass dependence. This is as opposed to the strong
# mass dependence of n_sh in equation (41)" -- so no choice of disc radius,
# scale height or clumping can make the disc criterion reproduce eq. 62.
#
# shell_density() implements that chain from SAGE26's OWN accretion rate,
# virial radius and virial velocity, rather than from D23's analytic eq. 31
# Mdot_ac / eq. 23-24 Rv, Vv fitting formulae. Fed D23's own analytic inputs
# it reproduces eq. 62 to within 0.04 dex, so any departure seen when it is
# fed SAGE26's quantities instead is a real difference between SAGE26's
# accretion history and D23's assumed one -- which is the whole point of
# computing it here rather than just redrawing eq. 62.
STREAM_RADIUS_FRACTION = 0.05   # R_str / Rvir, D23's fiducial (their eq. 61)
# Sound speed of the pre-shock stream gas. D23 take the post-shock gas to
# cool rapidly to T ~ 10^4 K (the T_4 term in eq. 41); 13 km/s reproduces
# their quoted Mach ~ 15 at V_v ~ 200 km/s. NOTE this is an assertion of
# D23's rapid-cooling assumption, not something SAGE26 computes: SAGE26
# routes infalling gas to the HOT reservoir at T_vir (add_infall_to_hot()
# in model_infall.c).
SHELL_SOUND_SPEED_KMS = 13.0


def br06_scale_height_cm(diskradius_cm):
    """Blitz & Rosolowsky (2006) eq. (9) *stellar* disc scale height,
    exactly as calculate_stellar_scale_height_BR06() computes it in
    model_h2_chemistry.c: log10(h*) = -0.23 + 0.8*log10(R*), both in pc,
    with R* the disc's exponential scale radius (DiskScaleRadius -- SAGE26
    passes it in as rs_pc with no re_factor rescaling, so this function does
    the same regardless of the re_factor chosen for the density volume's own
    radius). This is an empirical fit to the *stellar* disc for the BR06 H2
    midplane-pressure recipe, not a gas-disc scale height (see the module
    docstring) -- offered here as an alternative volume-averaging choice to
    explore, not a claim that it is the right one for cold gas.
    """
    r_pc = diskradius_cm / PC_TO_CM
    with np.errstate(divide='ignore', invalid='ignore'):
        log_h_pc = -0.23 + 0.8 * np.log10(r_pc)
    h_pc = np.where(r_pc > 0.0, 10.0 ** log_h_pc, 0.0)
    return h_pc * PC_TO_CM


DEFAULT_DIR = './output/microuchuu_noffb/'
MIN_MSTAR = 1.0e8          # Msun; drop numerically noisy, near-empty galaxies

# Bullock spin parameter, exactly as get_disk_radius() computes it in
# model_misc.c: lambda' = |Spin| / (SQRT_REPLACEMENT * Vvir * Rvir). Spin is
# stored in the same code units as Vvir*Rvir ([Mpc/h][km/s]), so the ratio
# needs no further unit conversion here. 1.414 (not sqrt(2)) is intentional --
# see the comment on SQRT_REPLACEMENT in model_misc.c.
SQRT_REPLACEMENT = 1.414

# Galaxies below this percentile of the spin-parameter distribution can be
# dropped before plotting: DiskScaleRadius = (lambda'/SQRT_REPLACEMENT) * Rvir
# has no floor, so the low-spin tail produces arbitrarily small discs and,
# via n ~ M/R^3, spuriously huge densities unrelated to FFB physics (see the
# module docstring). Off by default (0) -- pass --min-spin-percentile to
# apply it; 1.0 would drop the bottom 1%.
MIN_SPIN_PERCENTILE = 0.0

# Minimum N-body particle count (Len) for a halo's spin vector -- and hence
# its DiskScaleRadius and n, both derived from it -- to be trusted. Off by
# default (0) -- pass --min-particles to apply it; 32 is a standard rough
# minimum for trusting halo structural quantities in an N-body simulation.
# See read_snapshot()'s docstring for why this matters most at the earliest,
# most poorly-resolved output snapshot.
MIN_PARTICLES = 0


def spin_parameter(spinx, spiny, spinz, vvir, rvir):
    """Bullock spin parameter lambda', matching get_disk_radius() exactly."""
    spin_mag = np.sqrt(spinx**2 + spiny**2 + spinz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return spin_mag / (SQRT_REPLACEMENT * vvir * rvir)


# FIRE (Muratov et al. 2015) critical circular velocity separating the two
# reheating power-law slopes, exactly matching FIRE_V_CRIT_KMS in
# model_starformation_and_feedback.c.
FIRE_V_CRIT_KMS = 60.0

# The two FIRE mass-loading slopes below/above FIRE_V_CRIT_KMS, exactly
# matching compute_sn_feedback()'s fire_scaling term in
# model_starformation_and_feedback.c (Muratov+15 eq. 9/11).
FIRE_V_SLOPE_LOW = -3.2
FIRE_V_SLOPE_HIGH = -1.0


def fire_v_term(vvir):
    """FIRE (Muratov+15) velocity term: the same broken power law in Vvir
    as SAGE26's own SN mass-loading recipe -- (Vc/60)^-3.2 below the 60 km/s
    critical velocity, (Vc/60)^-1.0 above, Vc floored at 1 km/s -- matching
    compute_sn_feedback()'s fire_scaling exactly (the (1+z)^alpha factor is
    applied separately, by clumping_fire()).
    """
    vc = np.maximum(vvir, 1.0)
    slope = np.where(vc < FIRE_V_CRIT_KMS, FIRE_V_SLOPE_LOW, FIRE_V_SLOPE_HIGH)
    return (vc / FIRE_V_CRIT_KMS) ** slope

CMAP = plt.get_cmap('viridis')


def model_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'No model_*.hdf5 in {directory}')
    return files


def header(directory):
    """Cosmology and unit conversions, from the run's own output."""
    with h5py.File(model_files(directory)[0], 'r') as f:
        sim, runtime = f['Header/Simulation'], f['Header/Runtime']
        h = float(sim.attrs['hubble_h'])
        unit_mass_g = float(runtime.attrs['UnitMass_in_g'])
        unit_length_cm = float(runtime.attrs['UnitLength_in_cm'])
        return {
            'hubble_h': h,
            'to_msun': unit_mass_g / MSUN_G / h,      # code mass -> Msun
            'to_cm': unit_length_cm / h,               # code length -> cm
            'omega_m': float(sim.attrs['omega_matter']),
            'omega_l': float(sim.attrs['omega_lambda']),
            'baryon_frac': float(runtime.attrs['BaryonFrac']),
            'z': np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps': list(f['Header/output_snapshots'][:]),
        }


def cosmic_time_gyr(z, omega_m, omega_l, hubble_h):
    """Age of a flat LambdaCDM universe at redshift z [Gyr].

    t(z) = 2/(3 H0 sqrt(OmL)) * asinh[ sqrt(OmL/OmM) (1+z)^-3/2 ], the exact
    flat matter+Lambda solution -- needed only to turn the snapshot spacing
    into a dt for the tree-derived accretion rate (see gather_population()'s
    with_accretion), so a closed form beats pulling in astropy.
    """
    hubble_time_gyr = 9.778 / hubble_h          # 1/H0 in Gyr for H0 = 100h
    return (2.0 / (3.0 * np.sqrt(omega_l))) * hubble_time_gyr * \
        np.arcsinh(np.sqrt(omega_l / omega_m) * (1.0 + z) ** -1.5)


def read_snapshot(directory, hdr, snap, min_mstar, min_particles=0):
    """Galaxies' StellarMass, Mvir, ColdGas, disc radius, Vvir, spin parameter
    and FFBRegime, in physical units (spin parameter and FFBRegime are
    dimensionless/categorical).

    ColdGas, Mvir and DiskRadius are converted here rather than left in code
    units: the density calculation needs grams and centimetres directly.
    FFBRegime is SAGE26's own per-galaxy FFB flag (0/1, set by
    determine_and_store_ffb_regime() in model_regimes.c whenever
    FeedbackFreeModeOn > 0 -- always 0 on a FeedbackFreeModeOn=0 run), kept
    here so a second, FFB-enabled run's own selection can be compared
    directly against this script's own density/mass criteria -- see
    --ffb-regime-dir.

    min_particles (0 disables) drops haloes resolved with too few N-body
    particles (Len) to trust their spin vector -- and hence DiskScaleRadius
    and n, both derived from it. This matters most at the earliest output
    snapshot: on microuchuu_noffb, z=13.95 (its first snapshot) has every
    halo resolved with Len in [4, 27] (median ~15), against far higher
    counts by z<=8.5, and its shot-noise-dominated spin vectors drive
    spuriously small discs and hence spurious n spikes at that one snapshot
    -- see the module docstring and MIN_SPIN_PERCENTILE.
    """
    ms_list, mv_list, cg_list, r_list, vvir_list, spin_list, ffb_list = \
        [], [], [], [], [], [], []
    rvir_list, gid_list = [], []
    for path in model_files(directory):
        with h5py.File(path, 'r') as f:
            grp = f.get(f'Snap_{snap}')
            if grp is None:
                continue
            # HDF5 stores these as float32; the physical (non-code-unit) masses
            # and cgs radii overflow float32's ~3.4e38 range, so promote first.
            ms = np.asarray(grp['StellarMass'], dtype=np.float64) * hdr['to_msun']
            keep = ms >= min_mstar
            if min_particles > 0:
                keep &= np.asarray(grp['Len']) >= min_particles
            mv = np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun']
            cg = np.asarray(grp['ColdGas'], dtype=np.float64) * hdr['to_msun']
            r = np.asarray(grp['DiskRadius'], dtype=np.float64) * hdr['to_cm']
            rvir = np.asarray(grp['Rvir'], dtype=np.float64)      # Mpc/h, code units
            vvir = np.asarray(grp['Vvir'], dtype=np.float64)      # km/s
            spinx = np.asarray(grp['Spinx'], dtype=np.float64)
            spiny = np.asarray(grp['Spiny'], dtype=np.float64)
            spinz = np.asarray(grp['Spinz'], dtype=np.float64)
            spin = spin_parameter(spinx, spiny, spinz, vvir, rvir)
            ffb = np.asarray(grp['FFBRegime'], dtype=np.int32)
            keep &= (mv > 0) & (cg > 0) & (r > 0) & (vvir > 0) & np.isfinite(spin)
            ms_list.append(ms[keep])
            mv_list.append(mv[keep])
            cg_list.append(cg[keep])
            r_list.append(r[keep])
            vvir_list.append(vvir[keep])
            spin_list.append(spin[keep])
            ffb_list.append(ffb[keep])
            rvir_list.append(rvir[keep] * hdr['to_cm'])
            gid_list.append(np.asarray(grp['GalaxyIndex'], dtype=np.int64)[keep])
    return (np.concatenate(ms_list) if ms_list else np.array([]),
            np.concatenate(mv_list) if mv_list else np.array([]),
            np.concatenate(cg_list) if cg_list else np.array([]),
            np.concatenate(r_list) if r_list else np.array([]),
            np.concatenate(vvir_list) if vvir_list else np.array([]),
            np.concatenate(spin_list) if spin_list else np.array([]),
            np.concatenate(ffb_list) if ffb_list else np.array([], dtype=np.int32),
            np.concatenate(rvir_list) if rvir_list else np.array([]),
            np.concatenate(gid_list) if gid_list else np.array([], dtype=np.int64))


def progenitor_mvir(directory, hdr, snap):
    """(GalaxyIndex, Mvir [Msun]) for every galaxy at snapshot `snap`, sorted
    by GalaxyIndex so a caller can searchsorted into it.

    Unfiltered on purpose: a galaxy's progenitor need not itself pass the
    stellar-mass or resolution cuts the descendant passed, and dropping it
    would silently turn a real accretion rate into a missing one.

    GalaxyIndex is stable across snapshots in SAGE26's output (~97% of
    galaxies at one snapshot reappear at the next), which is what makes the
    tree-derived accretion rate in gather_population() possible without
    walking the merger tree itself.
    """
    gids, mvirs = [], []
    for path in model_files(directory):
        with h5py.File(path, 'r') as f:
            grp = f.get(f'Snap_{snap}')
            if grp is None:
                continue
            gids.append(np.asarray(grp['GalaxyIndex'], dtype=np.int64))
            mvirs.append(np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun'])
    if not gids:
        return np.array([], dtype=np.int64), np.array([])
    gid = np.concatenate(gids)
    mvir = np.concatenate(mvirs)
    order = np.argsort(gid)
    return gid[order], mvir[order]


def accretion_rate(directory, hdr, snap, gid, mvir_msun):
    """Baryonic accretion rate [Msun/Gyr] for the galaxies (gid, mvir_msun)
    at snapshot `snap`, from the halo's own mass growth since snapshot-1:

        Mdot_ac = BaryonFrac * (Mvir(snap) - Mvir(snap-1)) / dt

    This is SAGE26's OWN accretion history, not D23's analytic eq. 31
    fitting formula -- see shell_density()'s docstring for why that
    distinction is the entire point of computing this.

    NaN where the galaxy has no progenitor at snap-1 (newly resolved) or the
    halo lost mass (stripped satellites), so such galaxies simply drop out
    of the shell-density panel rather than contaminating it.
    """
    if snap - 1 < 0:
        return np.full(gid.size, np.nan)
    prog_gid, prog_mvir = progenitor_mvir(directory, hdr, snap - 1)
    if prog_gid.size == 0:
        return np.full(gid.size, np.nan)

    idx = np.searchsorted(prog_gid, gid)
    idx_clipped = np.clip(idx, 0, prog_gid.size - 1)
    matched = prog_gid[idx_clipped] == gid
    mvir_prev = np.where(matched, prog_mvir[idx_clipped], np.nan)

    dt_gyr = (cosmic_time_gyr(hdr['z'][snap], hdr['omega_m'], hdr['omega_l'], hdr['hubble_h'])
              - cosmic_time_gyr(hdr['z'][snap - 1], hdr['omega_m'], hdr['omega_l'],
                                hdr['hubble_h']))
    if dt_gyr <= 0:
        return np.full(gid.size, np.nan)
    return hdr['baryon_frac'] * (mvir_msun - mvir_prev) / dt_gyr


def gather_population(directory, hdr, zmin, zmax, min_mstar, min_spin_percentile,
                       min_particles=0, with_accretion=False):
    """Every galaxy from every output snapshot in [zmin, zmax]:
    per-galaxy z, StellarMass, log10(Mvir), cold-gas density n (the default
    re_factor=1/height_factor=DISC_HEIGHT_FACTOR/clumping=1 choice), the raw
    ColdGas [Msun] and DiskRadius [cm] the comparison figure recomputes n
    from under other volume-averaging choices, Vvir [km/s] (for
    --clumping-evolve-with vvir), FFBRegime (SAGE26's own per-galaxy FFB
    flag -- see read_snapshot()), Rvir [cm] and the baryonic accretion rate
    Mdot_ac [Msun/Gyr] (both for shell_density()), with the low-spin outlier
    tail (see MIN_SPIN_PERCENTILE) and poorly-resolved haloes (see
    min_particles) excluded.

    with_accretion=False (the default) returns Mdot_ac as all-NaN without
    reading anything extra: deriving it costs one additional snapshot read
    per snapshot (see accretion_rate()), which only the shell-density panel
    of figure_clumping_law_grid() actually needs.

    Every figure is built from this single pass so the cuts -- and the
    resulting galaxy sample -- are identical between them.
    """
    snaps = [s for s in hdr['output_snaps'] if zmin <= hdr['z'][s] <= zmax]
    z_list, ms_list, logmv_list, n_list, spin_list = [], [], [], [], []
    cg_list, r_list, vvir_list, ffb_list, rvir_list, mdot_list = [], [], [], [], [], []
    for snap in snaps:
        ms, mv, cg, r, vvir, spin, ffb, rvir, gid = read_snapshot(
            directory, hdr, snap, min_mstar, min_particles)
        if mv.size == 0:
            continue
        n = cold_gas_density(cg, r)
        z_list.append(np.full(mv.size, hdr['z'][snap]))
        ms_list.append(ms)
        logmv_list.append(np.log10(mv))
        n_list.append(n)
        spin_list.append(spin)
        cg_list.append(cg)
        r_list.append(r)
        vvir_list.append(vvir)
        ffb_list.append(ffb)
        rvir_list.append(rvir)
        mdot_list.append(accretion_rate(directory, hdr, snap, gid, mv) if with_accretion
                         else np.full(mv.size, np.nan))
    if not z_list:
        return (np.array([]),) * 10

    z = np.concatenate(z_list)
    ms = np.concatenate(ms_list)
    logmv = np.concatenate(logmv_list)
    n = np.concatenate(n_list)
    spin = np.concatenate(spin_list)
    cg = np.concatenate(cg_list)
    r = np.concatenate(r_list)
    vvir = np.concatenate(vvir_list)
    ffb = np.concatenate(ffb_list)
    rvir = np.concatenate(rvir_list)
    mdot = np.concatenate(mdot_list)

    if min_spin_percentile > 0:
        spin_floor = np.percentile(spin, min_spin_percentile)
        keep = spin >= spin_floor
        print(f'  excluding {np.count_nonzero(~keep)}/{keep.size} galaxies below the '
              f'{min_spin_percentile:g}th spin-parameter percentile '
              f"(lambda' < {spin_floor:.4f})")
        z, ms, logmv, n, cg, r, vvir, ffb, rvir, mdot = (
            z[keep], ms[keep], logmv[keep], n[keep], cg[keep], r[keep], vvir[keep],
            ffb[keep], rvir[keep], mdot[keep])

    return z, ms, logmv, n, cg, r, vvir, ffb, rvir, mdot


def cold_gas_density(coldgas_g_msun, diskradius_cm, re_factor=1.0,
                      height_factor=DISC_HEIGHT_FACTOR, clumping=1.0,
                      br06_height=False):
    """Mean cold-gas number density within radius re_factor*DiskRadius [cm^-3].

    n = clumping * ColdGas / [pi R^2 H] / (mu m_p): the disc is modelled as a
    cylinder of radius R = re_factor * DiskRadius (SAGE26's exponential scale
    radius; re_factor=1.0 uses it directly, matching FeedbackFreeModeOn=8 in
    model_regimes.c, or e.g. 1.68 converts to Dekel+23's own effective-radius
    convention, eq. 37) and height H = height_factor * R (the thin-disc
    default), rather than as a uniform sphere (cf. Dekel+23 eq. 37, which
    uses Re^3 for a spherical effective volume). br06_height=True replaces H
    with br06_scale_height_cm(DiskRadius) instead -- see that function's
    docstring for why this is a stellar, not gas, scale height. clumping
    boosts the disc-averaged density up to the density of the actual
    star-forming clumps (Dekel+23 sec. 6-7), matching FFBCloudClumping in
    model_regimes.c.
    """
    mass_g = clumping * coldgas_g_msun * MSUN_G
    r_cm = re_factor * diskradius_cm
    height_cm = br06_scale_height_cm(diskradius_cm) if br06_height else height_factor * r_cm
    volume_cm3 = np.pi * r_cm**2 * height_cm
    rho = mass_g / volume_cm3
    return rho / (MU_NEUTRAL * PROTON_MASS_G)


def shell_density(mdot_msun_per_gyr, rvir_cm, vvir_kms, clumping=1.0,
                   stream_fraction=STREAM_RADIUS_FRACTION,
                   sound_speed_kms=SHELL_SOUND_SPEED_KMS):
    """Dekel+23 post-shock shell number density n_sh [cm^-3], eqs. 38-41.

    The accreting gas is funnelled through a stream of radius
    R_str = stream_fraction * Rvir at the virial velocity, so its pre-shock
    density follows from mass conservation (eq. 39),

        rho_str = Mdot_ac / (pi R_str^2 V_v)

    and is then compressed at the shock by the Mach number squared (eq. 38,
    with M = V_v / c_s and c_s set by the post-shock cooled temperature --
    the T_4 term in eq. 41), giving

        n_sh = clumping * rho_str * M^2 / (mu m_p)

    UNLIKE cold_gas_density(), this does not use ColdGas at all: the shell
    density is an inflow-FLUX density, not a reservoir density. That is
    exactly where eq. 62's halo-mass dependence comes from -- Mdot_ac ~ M^1.14
    funnelled through R_str^2 ~ M^2/3, times M^2 ~ V_v^2 ~ M^2/3, giving
    n_sh ~ M^1.48 (1+z)^3 (eq. 41) and, after R_str = 0.05 Rv is substituted,
    n_sh ~ M^0.81 (1+z)^5 (eq. 61). Setting n_sh > n_fbk then gives eq. 62,
    whose -6.2 exponent is just -5/0.81.

    mdot_msun_per_gyr may be negative (a halo that lost mass between
    snapshots, e.g. a stripped satellite); those return NaN rather than a
    negative density.
    """
    mdot = np.where(np.asarray(mdot_msun_per_gyr) > 0.0, mdot_msun_per_gyr, np.nan)
    mdot_g_per_s = mdot * MSUN_G / GYR_TO_S
    r_str_cm = stream_fraction * rvir_cm
    v_cm_s = vvir_kms * KM_TO_CM
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_str = mdot_g_per_s / (np.pi * r_str_cm**2 * v_cm_s)
        mach_sq = (v_cm_s / (sound_speed_kms * KM_TO_CM)) ** 2
        return clumping * rho_str * mach_sq / (MU_NEUTRAL * PROTON_MASS_G)


def density_config_label(re_factor, height_factor, br06_height, clumping, clumping_label=None):
    """Shared, human-readable summary of a cold_gas_density() call's own
    re_factor/height_factor/br06_height/clumping choice, so every figure
    built from that choice can show which combination actually produced it
    (rather than only the comparison figure showing this)."""
    height_text = r'\mathrm{BR06}\ h_\star(R_d)' if br06_height else rf'{height_factor:g}\,R'
    clumping_text = clumping_label if clumping_label is not None else f'{clumping:g}'
    return rf'$R = {re_factor:g}\,R_d$, $H = {height_text}$, clumping $= {clumping_text}$'


def shell_config_label(stream_fraction, sound_speed_kms, clumping):
    """The shell_density() counterpart of density_config_label(): names the
    R_str/Rvir, sound-speed and clumping choice behind a --shell figure, so
    a shell figure states its configuration the same way a disc one does."""
    return (r'D23 shell $n_{\rm sh}$ (eqs. 38--41): '
            rf'$R_{{\rm str}} = {stream_fraction:g}\,R_{{\rm vir}}$, '
            rf'$c_s = {sound_speed_kms:g}$ km/s, clumping $= {clumping:g}$')


def figure_mass_redshift_plane(z, logmv, n, outdir, config_label=None,
                                density_label=r'Cold-gas number density, $n\ (\mathrm{cm}^{-3})$',
                                filename='MvirRedshiftPlane_ColdGasDensity.pdf'):
    """Halo-mass/redshift plane: every galaxy, coloured by its own
    cold-gas density, against Dekel+23's actual FFB threshold curve in this
    plane (eq. 62): M_v,ffb(z) = 10^10.8 Msun * ((1+z)/10)^-6.2.

    n is whatever re_factor/height_factor/br06_height/clumping combination
    the caller computed it with (see cold_gas_density()/density_config_label());
    config_label, if given, is shown in the title so the figure states which
    combination produced it.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Clip the colour scale at n_fbk itself, not a data percentile: n spans
    # ~7 dex here (a few rare, very compact discs at fixed ColdGas reach
    # n >~ 1e5), and using the data max as vmax would compress the bulk of
    # the population into a narrow, washed-out-looking colour band (and using
    # a percentile like the 99th still left vmax a factor few below n_fbk,
    # so "brightest yellow" did not actually mean "crosses n_fbk" -- most
    # maxed-out points were still short of it). Anchoring vmax = n_fbk means
    # saturated yellow really does mean "at or above threshold", consistent
    # with the red ring markers below.
    vmin = np.nanpercentile(n, 1)
    vmax = N_FBK_CM3
    order = np.argsort(n)  # densest points drawn last, on top
    sc = ax.scatter(z[order], logmv[order], c=n[order], s=10, alpha=0.7,
                    cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
                    edgecolors='none', zorder=2)

    above = n >= N_FBK_CM3
    if np.any(above):
        ax.scatter(z[above], logmv[above], s=28, facecolors='none',
                  edgecolors='red', linewidths=1.0, zorder=4,
                  label=rf'$n \geq n_{{\rm fbk}}$ ({np.count_nonzero(above)} galaxies)')

    z_curve = np.linspace(z.min(), z.max(), 200)
    log_mthresh = np.log10(ffb_threshold_mass_msun(z_curve))
    ax.plot(z_curve, log_mthresh, color='red', ls='--', lw=2.5, zorder=3,
            label=r'Dekel+23 eq. 62: $M_{\rm v,ffb} = 10^{10.8}\,M_\odot\,[(1{+}z)/10]^{-6.2}$')
    ax.legend(loc='upper right', frameon=False, fontsize=11)

    cbar = fig.colorbar(sc, ax=ax, extend='both')
    cbar.set_label(density_label)
    cbar.ax.text(0.5, 1.02, r'$\geq n_{\rm fbk}$', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=11, color='0.3')

    if config_label is not None:
        ax.set_title(config_label, fontsize=11)

    ax.set_xlabel(r'Redshift, $z$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$')
    ax.set_ylim(logmv.min() - 0.1, logmv.max() + 0.1)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


def figure_ffb_criteria_comparison(z, logmv, n, outdir, n_bins_z, n_bins_mv,
                                    sigmoid_delta_dex, config_label,
                                    ffb_regime_z=None, ffb_regime_logmv=None,
                                    ffb_regime=None, ffb_regime_dir=None):
    """Two-panel comparison of the two FFB selections in the (z, Mvir) plane.

    Left panel: the fraction of galaxies in each (z, Mvir) cell whose own
    density n crosses n_fbk (Dekel+23 eq. 5). n is whatever the caller
    computed -- the disc cold-gas density (cold_gas_density()) or, under
    --shell, the post-shock shell density (shell_density()) -- with
    config_label naming that choice for the panel title. A per-galaxy,
    simulated-gas-structure-dependent criterion either way.

    Right panel: by default, the analytic Li+2024 eq. (3) sigmoid FFB
    probability at the same (z, Mvir), exactly as calculate_ffb_fraction()
    computes it in model_regimes.c (FeedbackFreeModeOn in {1, 6}) -- a
    smooth function of halo mass and redshift alone, with no dependence on
    any simulated gas quantity, so on its own it covers the full (z, Mvir)
    rectangle rather than only where SAGE26 actually places galaxies. It is
    masked here to the left panel's own (z, Mvir) footprint -- NaN wherever
    that panel's bin holds zero simulated galaxies -- so a blank patch never
    gets mistaken for "the density criterion disagrees here": it means
    neither panel has anything to say about that cell.

    If ffb_regime_z/ffb_regime_logmv/ffb_regime are given instead (see
    --ffb-regime-dir), the right panel becomes the empirical fraction of
    galaxies with FFBRegime==1 in each (z, Mvir) cell, from a SEPARATE run
    with FeedbackFreeModeOn > 0 -- i.e. what SAGE26 itself actually selected
    as FFB in a live run, using whichever FeedbackFreeModeOn prescription
    that run was configured with (not necessarily the Li+24 sigmoid --
    check that run's own FeedbackFreeModeOn), binned into the SAME
    (z, Mvir) grid and masked to its OWN footprint (that run's galaxy
    population need not sample the same (z, Mvir) cells as the density run).

    Both panels share a [0, 1] colour scale and the same Dekel+23 eq. 62 /
    Li+24 eq. 2 threshold curve, so the two selections can be compared
    directly: whether, and over what range of (z, Mvir), a given disc
    radius/height/clumping choice makes the density criterion track the
    mass-only (or actual SAGE26-selected) one is exactly the question
    re_factor/height_factor/br06_height/clumping let you explore (see
    --re-factor/--height-factor/--br06-height/--clumping), or that the shell
    density answers from a different quantity entirely (--shell).

    Galaxies whose n is not finite (under --shell, those with no usable
    tree-derived accretion rate) are dropped from both the numerator and
    the denominator, so a cell shows the fraction among galaxies the
    criterion can actually be evaluated for.
    """
    usable = np.isfinite(n)
    above = usable & (n >= N_FBK_CM3)

    z_edges = np.linspace(z.min(), z.max(), n_bins_z + 1)
    logmv_edges = np.linspace(logmv.min(), logmv.max(), n_bins_mv + 1)
    total, _, _ = np.histogram2d(z[usable], logmv[usable], bins=[z_edges, logmv_edges])
    hit, _, _ = np.histogram2d(z[above], logmv[above], bins=[z_edges, logmv_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        frac_n = np.where(total > 0, hit / total, np.nan)

    z_curve = np.linspace(z.min(), z.max(), 200)
    log_mthresh_curve = np.log10(ffb_threshold_mass_msun(z_curve))

    use_empirical = ffb_regime is not None and ffb_regime.size > 0

    if use_empirical:
        # Bin the FFB-enabled run's own galaxies into the SAME (z, Mvir)
        # grid as the left panel, masked to its OWN footprint (this run's
        # galaxies need not fill the same cells the density run's do).
        total_r, _, _ = np.histogram2d(ffb_regime_z, ffb_regime_logmv,
                                       bins=[z_edges, logmv_edges])
        is_ffb = ffb_regime == 1
        hit_r, _, _ = np.histogram2d(ffb_regime_z[is_ffb], ffb_regime_logmv[is_ffb],
                                     bins=[z_edges, logmv_edges])
        with np.errstate(invalid='ignore', divide='ignore'):
            frac_right = np.where(total_r > 0, hit_r / total_r, np.nan)
        z_edges_right, logmv_edges_right = z_edges, logmv_edges
    else:
        # Evaluate the analytic sigmoid on a grid fine enough to look smooth,
        # then mask each fine-grid point to whichever coarse (z, Mvir) bin it
        # falls in, so the masked-out region exactly matches the left panel's.
        n_fine = 300
        z_fine = np.linspace(z_edges[0], z_edges[-1], n_fine)
        logmv_fine = np.linspace(logmv_edges[0], logmv_edges[-1], n_fine)
        zg, mg = np.meshgrid(z_fine, logmv_fine, indexing='ij')  # shape (n_fine, n_fine)
        frac_right = ffb_sigmoid_fraction(mg, zg, sigmoid_delta_dex)

        zi = np.clip(np.digitize(z_fine, z_edges) - 1, 0, n_bins_z - 1)
        mi = np.clip(np.digitize(logmv_fine, logmv_edges) - 1, 0, n_bins_mv - 1)
        has_data = total > 0  # shape (n_bins_z, n_bins_mv)
        frac_right = np.where(has_data[np.ix_(zi, mi)], frac_right, np.nan)
        z_edges_right, logmv_edges_right = z_fine, logmv_fine

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), sharex=True, sharey=True)
    # The style sheet's figure.autolayout=True runs tight_layout() on every
    # draw/save, which doesn't know how to place a colorbar spanning both
    # axes (it isn't part of the subplot gridspec) and warns every time.
    # This figure lays itself out via bbox_inches='tight' at savefig instead.
    fig.set_tight_layout(False)

    pcm = ax1.pcolormesh(z_edges, logmv_edges, frac_n.T, cmap='viridis',
                          vmin=0, vmax=1, shading='flat')
    ax1.plot(z_curve, log_mthresh_curve, color='red', ls='--', lw=2.0, zorder=3)
    ax1.set_title(r'$\langle n\rangle \geq n_{\rm fbk}$' + '\n' + config_label, fontsize=11)
    ax1.set_xlabel(r'Redshift, $z$')
    ax1.set_ylabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$')

    shading = 'flat' if use_empirical else 'auto'
    ax2.pcolormesh(z_edges_right, logmv_edges_right, frac_right.T, cmap='viridis',
                   vmin=0, vmax=1, shading=shading)
    ax2.plot(z_curve, log_mthresh_curve, color='red', ls='--', lw=2.0, zorder=3,
             label=r'$M_{\rm v,ffb}(z)$ (eq. 62 / Li+24 eq. 2)')
    if use_empirical:
        ax2.set_title('SAGE26 FFBRegime==1 (actual selection)\n'
                      f'run: {ffb_regime_dir}', fontsize=11)
    else:
        ax2.set_title(f'Li+24 eq. 3 sigmoid, ' + r'$\Delta\log M = '
                      rf'{sigmoid_delta_dex:g}$ dex, masked to the left panel '
                      r'footprint', fontsize=11)
    ax2.set_xlabel(r'Redshift, $z$')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.8, fontsize=9)

    cbar = fig.colorbar(pcm, ax=[ax1, ax2], fraction=0.046, pad=0.02)
    cbar.set_label('FFB fraction / probability')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBCriteriaComparison_MvirRedshift.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


# The six clumping-law choices compared by figure_clumping_law_grid(), each a
# (title, kind, br06_height) triple. kind selects which cold_gas_density()
# clumping to use: 'fixed' takes a literal number (1.0, or --grid-fixed-
# clumping), 'redshift'/'vvir'/'fire' fit one of this module's own evolving-
# clumping laws (see fit_evolving_clumping()/fit_evolving_clumping_vvir()/
# fit_evolving_clumping_fire()) to the population being plotted. Every entry
# but the first two uses BR06 height, the more physically-motivated scale
# height (see --br06-height), once clumping is doing real work.
CLUMPING_GRID_CONFIGS = [
    ('No clumping\n(thin-disc height)', 'fixed', False),
    ('No clumping\n(BR06 height)', 'fixed', True),
    (None, 'fixed_value', True),  # title filled in with --grid-fixed-clumping
    ('Evolving clumping$(z)$\n(BR06 height)', 'redshift', True),
    (r'Evolving clumping$(V_{\rm vir})$' + '\n(BR06 height)', 'vvir', True),
    ('Evolving clumping (FIRE)\n(BR06 height)', 'fire', True),
    # Not a clumping law at all: D23's SHELL density (eqs. 38-41), the
    # quantity eq. 62 is actually built on, from SAGE26's own accretion
    # rate rather than their analytic eq. 31 -- see shell_density().
    ('D23 shell density\n' + r'$n_{\rm sh}$ (eqs. 38-41)', 'shell', False),
]


def figure_clumping_law_grid(directory, hdr, zmin, zmax, min_mstar, outdir, sigmoid_delta_dex,
                              n_bins_z, n_bins_mv, fixed_clumping, clumping_fit_min_n,
                              clumping_fit_z0, clumping_fit_v0, clumping_vvir_bins,
                              grid_min_particles, grid_min_spin_percentile):
    """12 density-criterion panels (see figure_ffb_criteria_comparison()'s
    left panel) -- six clumping-law choices (CLUMPING_GRID_CONFIGS) each
    evaluated on two populations, SAGE26's raw output and the same low-spin/
    poorly-resolved cut --min-spin-percentile/--min-particles apply
    elsewhere in this script -- against ONE shared Li+24 eq. 3 sigmoid panel
    (figure_ffb_criteria_comparison()'s right panel) spanning the top, the
    fixed reference every other panel is compared to.

    Every panel -- sigmoid included -- shares the SAME (z, Mvir) grid
    (n_bins_z x n_bins_mv, from --comparison-zbins/--comparison-mvbins),
    built from the raw population's own range (the superset, since the cut
    population only removes galaxies): the two-panel comparison figure
    evaluates its sigmoid panel on a separate, much finer grid than its
    density panel purely for visual smoothness, which makes the two panels
    impossible to compare cell-for-cell -- this figure deliberately does not
    do that, since comparing panels cell-for-cell is the entire point here.

    Every panel also uses re_factor = EFFECTIVE_RADIUS_FACTOR (1.68),
    matching SAGE26's own actual FeedbackFreeModeOn=8 density criterion in
    model_regimes.c, not the comparison figure's own re_factor=1.0 default.
    """
    re_factor = EFFECTIVE_RADIUS_FACTOR

    populations = {}
    for label, min_particles, min_spin_percentile, cut_desc in (
            ('raw', 0, 0.0, 'SAGE26 raw output'),
            ('cut', grid_min_particles, grid_min_spin_percentile,
             f'--min-particles {grid_min_particles}, --min-spin-percentile '
             f'{grid_min_spin_percentile:g}')):
        z, _ms, logmv, _n, cg, r, vvir, _ffb, rvir, mdot = gather_population(
            directory, hdr, zmin, zmax, min_mstar, min_spin_percentile, min_particles,
            with_accretion=True)
        if z.size == 0:
            raise SystemExit(f"No galaxies survive the '{label}' population's cuts "
                              '(min_particles={min_particles}, min_spin_percentile='
                              f'{min_spin_percentile:g}) for figure_clumping_law_grid().')
        populations[label] = dict(z=z, logmv=logmv, cg=cg, r=r, vvir=vvir, rvir=rvir,
                                  mdot=mdot, desc=cut_desc)
        n_acc = int(np.count_nonzero(np.isfinite(mdot) & (mdot > 0)))
        print(f"  grid '{label}' population: {z.size} galaxies, {n_acc} with a usable "
              f'tree-derived accretion rate ({100.0 * n_acc / max(z.size, 1):.1f}%)')

    # Shared grid for every panel -- built from the raw population, the
    # superset (see the docstring).
    z_raw, logmv_raw = populations['raw']['z'], populations['raw']['logmv']
    z_edges = np.linspace(z_raw.min(), z_raw.max(), n_bins_z + 1)
    logmv_edges = np.linspace(logmv_raw.min(), logmv_raw.max(), n_bins_mv + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    logmv_centers = 0.5 * (logmv_edges[:-1] + logmv_edges[1:])
    z_curve = np.linspace(z_edges[0], z_edges[-1], 200)
    log_mthresh_curve = np.log10(ffb_threshold_mass_msun(z_curve))

    total_raw, _, _ = np.histogram2d(z_raw, logmv_raw, bins=[z_edges, logmv_edges])
    zg, mg = np.meshgrid(z_centers, logmv_centers, indexing='ij')
    frac_sigmoid = ffb_sigmoid_fraction(mg, zg, sigmoid_delta_dex)
    frac_sigmoid = np.where(total_raw > 0, frac_sigmoid, np.nan)

    def frac_map(pop, kind, br06_height, fixed_value=None):
        """The density-criterion fraction map for one (population, clumping
        law) combination, on the shared grid above."""
        z, logmv, cg, r, vvir = pop['z'], pop['logmv'], pop['cg'], pop['r'], pop['vvir']

        if kind == 'shell':
            # A different criterion entirely, not a clumping variant: the
            # post-shock SHELL density eq. 62 is actually built on, from
            # SAGE26's own Mdot_ac/Rvir/Vvir. Galaxies with no usable
            # accretion rate (no progenitor, or a shrinking halo) are NaN
            # and are dropped from both the numerator and the denominator,
            # so the panel shows the fraction among galaxies it can
            # actually evaluate rather than counting unknowns as failures.
            # clumping = 1 deliberately: eq. 62 is n_sh > n_fbk at c = 1 in
            # this normalisation (eq. 41's own 2.3e3 coefficient is already
            # ~n_fbk), and feeding D23's analytic inputs through
            # shell_density() at c = 1 reproduces eq. 62 to 0.04 dex.
            n_sh = shell_density(pop['mdot'], pop['rvir'], vvir, clumping=1.0)
            usable = np.isfinite(n_sh)
            total, _, _ = np.histogram2d(z[usable], logmv[usable],
                                         bins=[z_edges, logmv_edges])
            hot = usable & (n_sh >= N_FBK_CM3)
            hit, _, _ = np.histogram2d(z[hot], logmv[hot], bins=[z_edges, logmv_edges])
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(total > 0, hit / total, np.nan)

        if kind in ('fixed', 'fixed_value'):
            clumping_arg = 1.0 if kind == 'fixed' else fixed_value
        elif kind == 'redshift':
            A, b, z0, *_ = fit_evolving_clumping(z, cg, r, re_factor, DISC_HEIGHT_FACTOR,
                                                 br06_height, clumping_fit_min_n, clumping_fit_z0)
            clumping_arg = clumping_powerlaw(z, A, b, z0)
        elif kind == 'vvir':
            A, b, v0, *_ = fit_evolving_clumping_vvir(
                vvir, cg, r, re_factor, DISC_HEIGHT_FACTOR, br06_height, clumping_fit_min_n,
                clumping_fit_v0, clumping_vvir_bins)
            clumping_arg = clumping_powerlaw_vvir(vvir, A, b, v0)
        else:  # 'fire'
            A, alpha, *_ = fit_evolving_clumping_fire(
                z, vvir, cg, r, re_factor, DISC_HEIGHT_FACTOR, br06_height, clumping_fit_min_n,
                clumping_vvir_bins)
            clumping_arg = clumping_fire(z, vvir, A, alpha)

        n = cold_gas_density(cg, r, re_factor=re_factor, height_factor=DISC_HEIGHT_FACTOR,
                             clumping=clumping_arg, br06_height=br06_height)
        above = n >= N_FBK_CM3
        total, _, _ = np.histogram2d(z, logmv, bins=[z_edges, logmv_edges])
        hit, _, _ = np.histogram2d(z[above], logmv[above], bins=[z_edges, logmv_edges])
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(total > 0, hit / total, np.nan)

    ncols = len(CLUMPING_GRID_CONFIGS)
    fig = plt.figure(figsize=(2.9 * ncols, 13.0), layout='constrained')
    gs = fig.add_gridspec(4, ncols, height_ratios=[1.0, 1.0, 1.0, 1.0])

    # Top: the shared sigmoid panel, spanning 2 rows and centred across the
    # middle four of six columns.
    top_lo, top_hi = 1, ncols - 1
    ax_top = fig.add_subplot(gs[0:2, top_lo:top_hi])
    pcm = ax_top.pcolormesh(z_edges, logmv_edges, frac_sigmoid.T, cmap='viridis',
                            vmin=0, vmax=1, shading='flat')
    ax_top.plot(z_curve, log_mthresh_curve, color='red', ls='--', lw=2.5, zorder=3,
                label=r'$M_{\rm v,ffb}(z)$ (eq. 62 / Li+24 eq. 2)')
    ax_top.set_title('Li+24 eq. 3 sigmoid, ' + rf'$\Delta\log M = {sigmoid_delta_dex:g}$ dex'
                     '\nmasked to the raw population footprint -- the fixed reference below',
                     fontsize=12)
    ax_top.set_xlabel(r'Redshift, $z$')
    ax_top.set_ylabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$')
    ax_top.legend(loc='lower left', frameon=True, framealpha=0.8, fontsize=9)

    axes = [ax_top]
    for row, pop_label in ((2, 'raw'), (3, 'cut')):
        pop = populations[pop_label]
        for col, (title, kind, br06_height) in enumerate(CLUMPING_GRID_CONFIGS):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            if kind == 'fixed_value':
                title = rf'Clumping $= {fixed_clumping:g}$' + '\n(BR06 height)'
                frac = frac_map(pop, kind, br06_height, fixed_value=fixed_clumping)
            else:
                frac = frac_map(pop, kind, br06_height)
            pcm = ax.pcolormesh(z_edges, logmv_edges, frac.T, cmap='viridis',
                                vmin=0, vmax=1, shading='flat')
            ax.plot(z_curve, log_mthresh_curve, color='red', ls='--', lw=1.2, zorder=3)
            ax.set_title(title, fontsize=8.5)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(pop['desc'], fontsize=8)
            if row == 3:
                ax.set_xlabel(r'$z$', fontsize=8)

    cbar = fig.colorbar(pcm, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label('FFB fraction / probability')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBClumpingLawGrid.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


def spearman_rho(x, y):
    """Spearman rank correlation, no scipy dependency (small samples, no
    tie-correction needed for this diagnostic)."""
    def rank(a):
        order = np.argsort(a)
        ranks = np.empty(len(a))
        ranks[order] = np.arange(len(a))
        return ranks
    return np.corrcoef(rank(x), rank(y))[0, 1]


def figure_density_mass_scatter(z, logmv, n, outdir, config_label):
    """Per-snapshot diagnostic: each galaxy's own n plotted directly against
    its own Mvir (no binning, no fraction) -- to see directly whether the
    <n> panel's lack of a clean gradient at high z (see the module docstring
    discussion) is because too few galaxies are sampled there, or because n
    genuinely doesn't track Mvir at fixed z for this run.

    n_fbk (eq. 5) is drawn as a horizontal line and M_v,ffb(z) (eq. 62 /
    Li+24 eq. 2) as a vertical line; each panel also reports N, the Spearman
    rank correlation between log(Mvir) and log(n), and the scatter in
    log(n) at fixed z. A weak/noisy correlation despite a healthy N points
    at scatter (e.g. spin-driven DiskRadius variation), not sample size,
    being the bottleneck -- something no choice of re_factor/height_factor/
    clumping/br06_height can fix, since they shift every galaxy at a given
    z by the same factor rather than adding a genuine Mvir dependence.

    n is whatever the caller computed -- the disc cold-gas density or, under
    --shell, the post-shock shell density -- and config_label names that
    choice in the suptitle. Non-finite n (under --shell, galaxies with no
    usable accretion rate) are dropped per panel.
    """
    z_unique = np.sort(np.unique(z))

    ncols = min(5, len(z_unique))
    nrows = int(np.ceil(len(z_unique) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows),
                              sharey=True)
    axes = np.atleast_1d(axes).ravel()
    fig.set_tight_layout(False)

    for i, z_val in enumerate(z_unique):
        ax = axes[i]
        sel = (z == z_val) & np.isfinite(n)
        lm, nn = logmv[sel], n[sel]

        ax.scatter(lm, nn, s=10, alpha=0.5, color='steelblue', edgecolors='none')
        ax.set_yscale('log')
        ax.axhline(N_FBK_CM3, color='0.15', ls='--', lw=1.2, zorder=3)
        ax.axvline(np.log10(ffb_threshold_mass_msun(z_val)), color='red', ls='--',
                   lw=1.2, zorder=3)

        info = f'N = {sel.sum()}'
        if sel.sum() >= 5 and np.ptp(lm) > 0:
            rho = spearman_rho(lm, np.log10(nn))
            scatter_dex = np.std(np.log10(nn))
            info += f'\n' + rf'$\rho = {rho:+.2f}$' + f'\nscatter = {scatter_dex:.2f} dex'
        ax.text(0.04, 0.04, info, transform=ax.transAxes, fontsize=7.5,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_title(f'$z = {z_val:.2f}$', fontsize=10)
        ax.tick_params(labelsize=8)

    for j in range(len(z_unique), len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$')
    fig.supylabel(r'Number density, $n\ (\mathrm{cm}^{-3})$')
    fig.suptitle(config_label, fontsize=11, y=1.0)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'DensityMassScatter_PerSnapshot.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


def figure_required_boost(z, cg_msun, r_cm, outdir, height_factor, br06_height):
    """Per-snapshot diagnostic: how much clumping, or how much disc-radius
    shrinkage, each galaxy's own raw density n0 (re_factor=1, clumping=1,
    this call's height_factor/br06_height convention) is short of n_fbk by.

    required_clumping = n_fbk / n0: the FFBCloudClumping-style boost that
    would push a galaxy at its actual DiskRadius up to n_fbk. A model-side
    fix -- it says nothing changes about the disc's size, only that more of
    its gas is imagined concentrated into the star-forming clumps.

    required_re_factor = (n0 / n_fbk)^(1/p): the disc-radius rescaling that
    would push the same galaxy up to n_fbk at clumping=1, where n ~ R^-p at
    fixed ColdGas (see cold_gas_density()) -- p=3 for the default
    height_factor*R convention (both R and H shrink together), or p=2 for
    br06_height (H is fixed, independent of R, since it depends only on the
    unscaled DiskRadius). A geometry-side fix -- it says the gas is where
    ColdGas/DiskRadius already put it, but concentrated into a smaller disc
    than SAGE26's own exponential scale radius suggests.

    Values < 1 mean the raw density already exceeds n_fbk (no boost
    needed); values > 1 quantify the shortfall. Plotting both per redshift
    (median with 16th-84th percentile band) makes it possible to read off
    directly whether the mass route or the geometry route is doing more
    work to close the gap, and how that balance shifts with z.
    """
    n0 = cold_gas_density(cg_msun, r_cm, re_factor=1.0, height_factor=height_factor,
                          clumping=1.0, br06_height=br06_height)
    required_clumping = N_FBK_CM3 / n0
    p = 2.0 if br06_height else 3.0
    required_re_factor = (n0 / N_FBK_CM3) ** (1.0 / p)

    z_unique = np.sort(np.unique(z))
    clump_med, clump_lo, clump_hi = [], [], []
    re_med, re_lo, re_hi = [], [], []
    for z_val in z_unique:
        sel = z == z_val
        clump_med.append(np.median(required_clumping[sel]))
        clump_lo.append(np.percentile(required_clumping[sel], 16))
        clump_hi.append(np.percentile(required_clumping[sel], 84))
        re_med.append(np.median(required_re_factor[sel]))
        re_lo.append(np.percentile(required_re_factor[sel], 16))
        re_hi.append(np.percentile(required_re_factor[sel], 84))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True)
    # Fixed margins rather than the style sheet's figure.autolayout=True: the
    # rotated multi-line ylabels below are wide enough that tight_layout's
    # automatic left margin clips them.
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.08, hspace=0.12)

    ax1.fill_between(z_unique, clump_lo, clump_hi, color='steelblue', alpha=0.25,
                     label='16th-84th percentile')
    ax1.plot(z_unique, clump_med, 'o-', color='steelblue', label='median')
    ax1.axhline(1.0, color='0.15', ls='--', lw=1.2, label=r'no boost needed ($n_0 \geq n_{\rm fbk}$)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Required clumping\n' + r'$n_{\rm fbk}/n_0$ at $R=R_d$', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.set_title('How much boost closes the gap to $n_{\\rm fbk}$: mass '
                  '(clumping) vs. geometry (disc radius)', fontsize=11)

    ax2.fill_between(z_unique, re_lo, re_hi, color='indianred', alpha=0.25)
    ax2.plot(z_unique, re_med, 'o-', color='indianred')
    ax2.axhline(1.0, color='0.15', ls='--', lw=1.2)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Redshift, $z$')
    height_note = r'BR06 $H$, $p=2$' if br06_height else rf'$H={height_factor:g}R$, $p=3$'
    ax2.set_ylabel('Required $R/R_d$\n' + r'$(n_0/n_{\rm fbk})^{1/p}$ at clumping$=1$',
                   fontsize=10)
    ax2.text(0.02, 0.04, height_note, transform=ax2.transAxes, fontsize=8,
             va='bottom', ha='left', color='0.3')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'RequiredBoost_vs_Redshift.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')


def clumping_powerlaw(z, A, b, z0):
    """Evolving clumping factor, clumping(z) = A * ((1+z)/(1+z0))^b."""
    return A * ((1.0 + z) / (1.0 + z0)) ** b


def fit_evolving_clumping(z, cg_msun, r_cm, re_factor, height_factor, br06_height,
                           min_n, z0):
    """Fit clumping(z) = A * ((1+z)/(1+z0))^b (see clumping_powerlaw()) to the
    per-snapshot MEDIAN clumping required to reach n_fbk at the given
    re_factor/height_factor/br06_height convention -- i.e. the same
    n_fbk/n0 diagnostic as figure_required_boost(), but evaluated at the
    actual (re_factor, height) choice this fit will be combined with,
    rather than that figure's fixed re_factor=1 baseline.

    Only snapshots with at least min_n galaxies enter the fit: the
    highest-z snapshots here can have as few as a handful of galaxies (see
    the module docstring / DensityMassScatter_PerSnapshot diagnostic), and
    a power law fit through those points would be dictated by small-number
    noise rather than a real trend. z0 is the pivot redshift, matching the
    eq. 62/Li+24 eq. 2 threshold curve's own z=9 pivot by default.

    Returns (A, b, z0, fit_z, fit_med, all_z, all_med, all_n): the fitted
    parameters, the (z, median) pairs actually used in the fit, and the
    same for every snapshot (fit or not) plus each snapshot's N, for
    reporting.
    """
    n_base = cold_gas_density(cg_msun, r_cm, re_factor=re_factor, height_factor=height_factor,
                              clumping=1.0, br06_height=br06_height)
    required = N_FBK_CM3 / n_base

    z_unique = np.sort(np.unique(z))
    all_z, all_med, all_n = [], [], []
    fit_z, fit_med = [], []
    for z_val in z_unique:
        sel = z == z_val
        med = np.median(required[sel])
        all_z.append(z_val)
        all_med.append(med)
        all_n.append(sel.sum())
        if sel.sum() >= min_n:
            fit_z.append(z_val)
            fit_med.append(med)

    fit_z, fit_med = np.array(fit_z), np.array(fit_med)
    if fit_z.size < 2:
        raise SystemExit(
            f'Only {fit_z.size} snapshot(s) have >= {min_n} galaxies -- not enough to fit '
            'an evolving clumping factor. Lower --clumping-fit-min-n or widen --zmin/--zmax.')

    x = np.log10((1.0 + fit_z) / (1.0 + z0))
    y = np.log10(fit_med)
    b, log10_A = np.polyfit(x, y, 1)
    A = 10.0 ** log10_A

    return A, b, z0, fit_z, fit_med, np.array(all_z), np.array(all_med), np.array(all_n)


def clumping_powerlaw_vvir(vvir, A, b, v0):
    """Evolving clumping factor, clumping(Vvir) = A * (Vvir/v0)^b.

    Virial density scales as Vvir^2 at fixed radius, so Vvir is a halo
    property that tracks the same "how dense was this system" driver as
    redshift, but per-halo rather than per-epoch: two haloes at the same z
    with different Vvir (different concentration/formation time) get
    different clumping instead of the same one.
    """
    return A * (vvir / v0) ** b


def fit_evolving_clumping_vvir(vvir, cg_msun, r_cm, re_factor, height_factor, br06_height,
                                min_n, v0, n_bins):
    """Fit clumping(Vvir) = A * (Vvir/v0)^b (see clumping_powerlaw_vvir()) to
    the per-bin MEDIAN clumping required to reach n_fbk, at the given
    re_factor/height_factor/br06_height convention -- the Vvir analogue of
    fit_evolving_clumping().

    Vvir varies continuously within a snapshot (unlike z, which is one value
    per snapshot), so galaxies are grouped into n_bins log-spaced Vvir bins
    spanning the sample's own range rather than by snapshot. Only bins with
    at least min_n galaxies enter the fit, for the same reason as
    fit_evolving_clumping(): a handful of galaxies in the extreme bins would
    otherwise dominate the fit with shot noise rather than a real trend.

    Returns (A, b, v0, fit_v, fit_med, all_v, all_med, all_n): the fitted
    parameters, the (Vvir bin centre, median) pairs actually used in the
    fit, and the same for every populated bin (fit or not) plus each bin's
    N, for reporting.
    """
    n_base = cold_gas_density(cg_msun, r_cm, re_factor=re_factor, height_factor=height_factor,
                              clumping=1.0, br06_height=br06_height)
    required = N_FBK_CM3 / n_base

    log_v = np.log10(vvir)
    edges = np.linspace(log_v.min(), log_v.max(), n_bins + 1)
    which = np.clip(np.digitize(log_v, edges) - 1, 0, n_bins - 1)

    all_v, all_med, all_n = [], [], []
    fit_v, fit_med = [], []
    for i in range(n_bins):
        sel = which == i
        if not np.any(sel):
            continue
        v_center = 10.0 ** (0.5 * (edges[i] + edges[i + 1]))
        med = np.median(required[sel])
        all_v.append(v_center)
        all_med.append(med)
        all_n.append(int(sel.sum()))
        if sel.sum() >= min_n:
            fit_v.append(v_center)
            fit_med.append(med)

    fit_v, fit_med = np.array(fit_v), np.array(fit_med)
    if fit_v.size < 2:
        raise SystemExit(
            f'Only {fit_v.size} Vvir bin(s) have >= {min_n} galaxies -- not enough to fit '
            'an evolving clumping factor. Lower --clumping-fit-min-n or --clumping-vvir-bins.')

    x = np.log10(fit_v / v0)
    y = np.log10(fit_med)
    b, log10_A = np.polyfit(x, y, 1)
    A = 10.0 ** log10_A

    return A, b, v0, fit_v, fit_med, np.array(all_v), np.array(all_med), np.array(all_n)


def clumping_fire(z, vvir, A, alpha):
    """Evolving clumping factor using the SAME functional form as SAGE26's
    own FIRE (Muratov+15) supernova mass-loading scaling, fire_scaling in
    compute_sn_feedback() (model_starformation_and_feedback.c):

        clumping(z, Vvir) = A * (1+z)^alpha * fire_v_term(Vvir)

    i.e. a power law in (1+z) times the same broken power law in Vvir the SN
    recipe uses (see fire_v_term()) -- alpha plays the role of
    RedshiftPowerLawExponent, but is fit to the required-clumping data here
    rather than reused from the SN recipe's own calibration, since there is
    no reason the two physical effects should share a normalization or
    redshift slope, only the same shape.
    """
    return A * (1.0 + z) ** alpha * fire_v_term(vvir)


def fit_evolving_clumping_fire(z, vvir, cg_msun, r_cm, re_factor, height_factor, br06_height,
                                min_n, n_vvir_bins):
    """Fit clumping(z, Vvir) = A * (1+z)^alpha * fire_v_term(Vvir) (see
    clumping_fire()) to the per-(snapshot, Vvir bin) MEDIAN clumping required
    to reach n_fbk, at the given re_factor/height_factor/br06_height
    convention.

    Only A and alpha are fit: the Vvir dependence's shape (break at
    FIRE_V_CRIT_KMS, slopes FIRE_V_SLOPE_LOW/HIGH) is fixed to SAGE26's own
    FIRE recipe, not fit, so this is a 1D log-log linear fit exactly like
    fit_evolving_clumping() -- y = log10(required) - log10(fire_v_term(Vvir))
    against x = log10(1+z) -- just with the Vvir term divided out first
    instead of Vvir ignored.

    Vvir is binned within each snapshot (n_vvir_bins log-spaced bins, as in
    fit_evolving_clumping_vvir()) since it varies continuously there; a bin
    (snapshot, Vvir-bin) pair enters the fit only with >= min_n galaxies.

    Returns (A, alpha, all_z, all_v, all_med, all_n, fit_mask): the fitted
    parameters, and every populated (snapshot, Vvir bin)'s centre/median/N
    plus a boolean mask of which entered the fit, for reporting.
    """
    n_base = cold_gas_density(cg_msun, r_cm, re_factor=re_factor, height_factor=height_factor,
                              clumping=1.0, br06_height=br06_height)
    required = N_FBK_CM3 / n_base

    z_unique = np.sort(np.unique(z))
    log_v = np.log10(np.maximum(vvir, 1.0))

    all_z, all_v, all_med, all_n, fit_mask = [], [], [], [], []
    fit_x, fit_y = [], []
    for z_val in z_unique:
        sel_z = z == z_val
        lv = log_v[sel_z]
        req_z = required[sel_z]
        edges = np.linspace(lv.min(), lv.max(), n_vvir_bins + 1)
        which = np.clip(np.digitize(lv, edges) - 1, 0, n_vvir_bins - 1)
        for i in range(n_vvir_bins):
            sel = which == i
            if not np.any(sel):
                continue
            v_center = 10.0 ** (0.5 * (edges[i] + edges[i + 1]))
            med = np.median(req_z[sel])
            n_bin = int(sel.sum())
            all_z.append(z_val)
            all_v.append(v_center)
            all_med.append(med)
            all_n.append(n_bin)
            used = n_bin >= min_n
            fit_mask.append(used)
            if used:
                fit_x.append(np.log10(1.0 + z_val))
                fit_y.append(np.log10(med) - np.log10(fire_v_term(v_center)))

    fit_x, fit_y = np.array(fit_x), np.array(fit_y)
    if fit_x.size < 2:
        raise SystemExit(
            f'Only {fit_x.size} (snapshot, Vvir bin) pair(s) have >= {min_n} galaxies -- not '
            'enough to fit an evolving clumping factor. Lower --clumping-fit-min-n or '
            '--clumping-vvir-bins.')

    alpha, log10_A = np.polyfit(fit_x, fit_y, 1)
    A = 10.0 ** log10_A

    return (A, alpha, np.array(all_z), np.array(all_v), np.array(all_med), np.array(all_n),
            np.array(fit_mask))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=DEFAULT_DIR, help=f'run directory (default: {DEFAULT_DIR})')
    ap.add_argument('--outdir', default=None, help='default: <dir>/plots/')
    ap.add_argument('--zmin', type=float, default=5.0, help='lowest redshift snapshot to include')
    ap.add_argument('--zmax', type=float, default=15.0, help='highest redshift snapshot to include')
    ap.add_argument('--min-mstar', type=float, default=MIN_MSTAR, help='Msun; 0 to disable')
    ap.add_argument('--min-spin-percentile', type=float, default=MIN_SPIN_PERCENTILE,
                     help='drop haloes below this percentile of the spin-parameter '
                          'distribution (0 to disable; default %(default)g)')
    ap.add_argument('--min-particles', type=int, default=MIN_PARTICLES,
                     help='drop haloes resolved with fewer than this many N-body '
                          'particles (Len), whose spin vector -- and hence DiskRadius '
                          'and n -- is too shot-noise-dominated to trust (0 to disable; '
                          'default %(default)d). Matters most at the earliest, most '
                          'poorly-resolved output snapshot -- see read_snapshot().')
    ap.add_argument('--re-factor', type=float, default=1.0,
                     help='disc radius for the <n> criterion in the comparison figure, '
                          'as a multiple of DiskRadius (1.68 for Dekel+23\'s own effective-'
                          'radius convention; default %(default)g)')
    ap.add_argument('--height-factor', type=float, default=DISC_HEIGHT_FACTOR,
                     help='disc height for the <n> criterion in the comparison figure, '
                          'as a multiple of the radius above; ignored if --br06-height is '
                          'set (default %(default)g)')
    ap.add_argument('--br06-height', action='store_true',
                     help='use the Blitz & Rosolowsky (2006) eq. 9 stellar disc scale '
                          'height h*(DiskRadius) -- exactly as calculate_stellar_scale_'
                          'height_BR06() computes it in model_h2_chemistry.c -- for the '
                          '<n> criterion in the comparison figure, instead of '
                          '--height-factor * R. Note this is an empirical fit to the '
                          '*stellar* disc, not a gas quantity (see the module docstring).')
    ap.add_argument('--clumping', type=float, default=1.0,
                     help='density clumping factor for the <n> criterion in the '
                          'comparison figure, matching FFBCloudClumping in '
                          'model_regimes.c (default %(default)g). Ignored if '
                          '--evolving-clumping is set.')
    ap.add_argument('--evolving-clumping', action='store_true',
                     help='use a fitted evolving clumping factor instead of the constant '
                          '--clumping, for the <n> criterion in the comparison and '
                          'density-mass-scatter figures -- see --clumping-evolve-with for '
                          'what it evolves with. Fit to the per-bin median clumping '
                          'required to reach n_fbk at the chosen --re-factor/--height-'
                          'factor/--br06-height, using only bins with >= '
                          '--clumping-fit-min-n galaxies.')
    ap.add_argument('--clumping-evolve-with', choices=('redshift', 'vvir', 'fire'),
                     default='redshift',
                     help="what --evolving-clumping fits the power law against: 'redshift' "
                          '-- clumping(z) = A*((1+z)/(1+z0))^b, one point per snapshot (see '
                          "fit_evolving_clumping()) -- 'vvir' -- clumping(Vvir) = "
                          'A*(Vvir/v0)^b, binned across the whole sample\'s own Vvir range '
                          "(see fit_evolving_clumping_vvir()) -- or 'fire' -- clumping(z, "
                          'Vvir) = A*(1+z)^alpha*fire_v_term(Vvir), the SAME functional '
                          "form as SAGE26's own FIRE (Muratov+15) SN mass-loading recipe "
                          '(fire_scaling in compute_sn_feedback(), '
                          'model_starformation_and_feedback.c) -- a broken power law in '
                          'Vvir with FIXED slopes/break taken from that recipe, times a '
                          '(1+z)^alpha term, with only A and alpha fit here (see '
                          'clumping_fire()/fit_evolving_clumping_fire()). Vvir is a '
                          'per-halo, not per-epoch, density proxy (virial density ~ Vvir^2 '
                          'at fixed radius), so vvir/fire capture halo-to-halo scatter at '
                          'fixed z that a redshift-only fit cannot (default %(default)s).')
    ap.add_argument('--clumping-fit-min-n', type=int, default=30,
                     help='minimum galaxies a redshift snapshot, Vvir bin, or (snapshot, '
                          'Vvir bin) pair needs to enter the --evolving-clumping fit '
                          '(default %(default)d)')
    ap.add_argument('--clumping-fit-z0', type=float, default=9.0,
                     help='pivot redshift for the --clumping-evolve-with redshift power '
                          'law, matching the eq. 62/Li+24 eq. 2 threshold curve convention '
                          '(default %(default)g)')
    ap.add_argument('--clumping-fit-v0', type=float, default=50.0,
                     help='pivot Vvir [km/s] for the --clumping-evolve-with vvir power law '
                          '(default %(default)g)')
    ap.add_argument('--clumping-vvir-bins', type=int, default=20,
                     help='number of log-spaced Vvir bins for the --clumping-evolve-with '
                          'vvir/fire fits -- for fire, binned within each snapshot '
                          '(default %(default)d)')
    ap.add_argument('--sigmoid-delta-dex', type=float, default=FFB_SIGMOID_DELTA_DEX,
                     help='half-width, in dex of Mvir, of the Li+24 eq. 3 sigmoid in the '
                          'comparison figure (default %(default)g)')
    ap.add_argument('--comparison-zbins', type=int, default=20,
                     help='number of redshift bins for the <n> panel of the comparison '
                          'figure (default %(default)d)')
    ap.add_argument('--comparison-mvbins', type=int, default=20,
                     help='number of log(Mvir) bins for the <n> panel of the comparison '
                          'figure (default %(default)d)')
    ap.add_argument('--ffb-regime-dir', default=None,
                     help='a SEPARATE run directory with FeedbackFreeModeOn > 0 (e.g. '
                          'output/microuchuu, as opposed to the density run\'s own '
                          'output/microuchuu_noffb). If given, the comparison figure\'s '
                          'right panel becomes the empirical fraction of that run\'s own '
                          'galaxies with FFBRegime==1 in each (z, Mvir) cell -- what SAGE26 '
                          'itself actually selected as FFB there -- instead of the analytic '
                          'Li+24 eq. 3 sigmoid. Uses this run\'s own --zmin/--zmax/--min-'
                          'mstar/--min-spin-percentile/--min-particles cuts.')
    ap.add_argument('--shell', action='store_true',
                     help="build every figure from Dekel+23's post-shock SHELL density "
                          'n_sh (eqs. 38-41, see shell_density()) instead of the disc '
                          'cold-gas density -- computed from SAGE26\'s own tree-derived '
                          'accretion rate, Rvir and Vvir, and using no ColdGas at all. '
                          'This is the quantity eq. 62 is actually built on (D23 sec. 8.2 '
                          'is explicit that the disc density is NOT), so it is the '
                          'like-for-like comparison against the eq. 62 curve. Ignores '
                          '--re-factor/--height-factor/--br06-height/--clumping/'
                          '--evolving-clumping, which are disc-geometry choices, and '
                          'skips the disc-only RequiredBoost figure.')
    ap.add_argument('--shell-stream-fraction', type=float, default=STREAM_RADIUS_FRACTION,
                     help="R_str/Rvir for --shell, D23's fiducial (default %(default)g)")
    ap.add_argument('--shell-sound-speed', type=float, default=SHELL_SOUND_SPEED_KMS,
                     metavar='KMS',
                     help='post-shock sound speed [km/s] for --shell, setting the Mach '
                          "compression; D23's T ~ 10^4 K gives ~13, i.e. Mach ~ 15 at "
                          'Vvir ~ 200 km/s (default %(default)g)')
    ap.add_argument('--shell-clumping', type=float, default=1.0,
                     help='clumping factor c for --shell. Note eq. 62 corresponds to c = 1 '
                          "in this normalisation (eq. 41's own coefficient is already "
                          '~n_fbk), so the default reproduces eq. 62 rather than '
                          'offsetting from it (default %(default)g)')
    ap.add_argument('--grid', action='store_true',
                     help='produce FFBClumpingLawGrid.pdf: a shared Li+24 eq. 3 sigmoid '
                          'panel at the top, compared against 12 density-criterion panels '
                          'below it -- CLUMPING_GRID_CONFIGS\' six clumping-law choices, '
                          'each on both the raw population and the --grid-min-particles/'
                          '--grid-min-spin-percentile cut one (see '
                          'figure_clumping_law_grid()). Every panel uses re_factor = '
                          'EFFECTIVE_RADIUS_FACTOR (1.68), independent of --re-factor.')
    ap.add_argument('--grid-fixed-clumping', type=float, default=6.0,
                     help='the constant FFBCloudClumping value for the --grid panel that '
                          'uses one, instead of an evolving law (default %(default)g)')
    ap.add_argument('--grid-min-particles', type=int, default=20,
                     help='--min-particles for the --grid figure\'s "cut" population row '
                          '(default %(default)d)')
    ap.add_argument('--grid-min-spin-percentile', type=float, default=1.0,
                     help='--min-spin-percentile for the --grid figure\'s "cut" population '
                          'row (default %(default)g)')
    args = ap.parse_args()

    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    hdr = header(args.dir)

    (z_all, ms_all, logmv_all, n_all, cg_all, r_all, vvir_all, _ffb_all, rvir_all,
     mdot_all) = gather_population(
        args.dir, hdr, args.zmin, args.zmax, args.min_mstar, args.min_spin_percentile,
        args.min_particles, with_accretion=args.shell)
    if z_all.size == 0:
        raise SystemExit('No galaxies passed the selection cuts in the requested redshift range.')
    z_unique = np.sort(np.unique(z_all))

    # --shell replaces the density every figure is built from: the post-shock
    # shell density, from SAGE26's own accretion rate, rather than the disc
    # cold-gas density. Every disc-geometry knob is inapplicable, so short-
    # circuit the whole clumping-law block rather than silently ignoring them.
    if args.shell:
        disc_flags = [name for name, used in (
            ('--re-factor', args.re_factor != 1.0),
            ('--height-factor', args.height_factor != DISC_HEIGHT_FACTOR),
            ('--br06-height', args.br06_height),
            ('--clumping', args.clumping != 1.0),
            ('--evolving-clumping', args.evolving_clumping)) if used]
        if disc_flags:
            print(f'  note: --shell ignores {", ".join(disc_flags)} '
                  '(disc-geometry options; the shell density uses no disc quantities)')
        n_all = shell_density(mdot_all, rvir_all, vvir_all,
                              clumping=args.shell_clumping,
                              stream_fraction=args.shell_stream_fraction,
                              sound_speed_kms=args.shell_sound_speed)
        usable = np.isfinite(n_all)
        print(f'  shell density: {np.count_nonzero(usable)}/{n_all.size} galaxies have a '
              f'usable tree-derived accretion rate '
              f'({100.0 * np.count_nonzero(usable) / max(n_all.size, 1):.1f}%)')
        if not np.any(usable):
            raise SystemExit('No galaxy has a usable accretion rate -- --shell needs at '
                             'least two consecutive snapshots in [--zmin, --zmax].')
        config_label = shell_config_label(args.shell_stream_fraction, args.shell_sound_speed,
                                          args.shell_clumping)
        clumping_arg, clumping_label = None, None

    # Resolve the actual re_factor/height_factor/br06_height/clumping combination
    # up front, and use it for EVERY figure below (figure_required_boost excepted,
    # which by design always isolates re_factor=1/clumping=1 as its neutral
    # baseline) -- so a plot never silently reverts to cold_gas_density()'s
    # defaults instead of what was actually asked for on the command line.
    elif args.evolving_clumping and args.clumping_evolve_with == 'fire':
        A, alpha, all_z, all_v, all_med, all_n, fit_mask = fit_evolving_clumping_fire(
            z_all, vvir_all, cg_all, r_all, args.re_factor, args.height_factor,
            args.br06_height, args.clumping_fit_min_n, args.clumping_vvir_bins)
        print(f'evolving clumping fit: clumping(z, Vvir) = {A:.3g} * (1+z)^{alpha:+.3g} * '
              f"fire_v_term(Vvir)  [SAGE26's own FIRE Vvir shape: break "
              f'{FIRE_V_CRIT_KMS:g} km/s, slopes {FIRE_V_SLOPE_LOW:g}/{FIRE_V_SLOPE_HIGH:g}; '
              f'fit to {int(fit_mask.sum())}/{fit_mask.size} (snapshot, Vvir bin) pairs with '
              f'>= {args.clumping_fit_min_n} galaxies]')
        for zv, vv, med, nn, used in zip(all_z, all_v, all_med, all_n, fit_mask):
            tag = 'fit' if used else 'excluded (N too small)'
            print(f'  z={zv:5.2f}  Vvir={vv:7.1f}  N={nn:5d}  median required clumping='
                  f'{med:7.2f}  fitted={clumping_fire(zv, vv, A, alpha):7.2f}  [{tag}]')
        clumping_arg = clumping_fire(z_all, vvir_all, A, alpha)
        clumping_label = (rf'{A:.2g}\,(1{{+}}z)^{{{alpha:+.2g}}}\times'
                          r'{\rm FIRE}(V_{\rm vir})')
    elif args.evolving_clumping and args.clumping_evolve_with == 'vvir':
        A, b, v0, fit_v, _fit_med, all_v, all_med, all_n = fit_evolving_clumping_vvir(
            vvir_all, cg_all, r_all, args.re_factor, args.height_factor, args.br06_height,
            args.clumping_fit_min_n, args.clumping_fit_v0, args.clumping_vvir_bins)
        print(f'evolving clumping fit: clumping(Vvir) = {A:.3g} * (Vvir/{v0:.0f})^{b:+.3g} '
              f'[fit to {fit_v.size}/{all_v.size} Vvir bins with >= {args.clumping_fit_min_n} '
              'galaxies]')
        for vv, med, nn in zip(all_v, all_med, all_n):
            used = 'fit' if nn >= args.clumping_fit_min_n else 'excluded (N too small)'
            print(f'  Vvir={vv:7.1f}  N={nn:5d}  median required clumping={med:7.2f}  '
                  f'fitted={clumping_powerlaw_vvir(vv, A, b, v0):7.2f}  [{used}]')
        clumping_arg = clumping_powerlaw_vvir(vvir_all, A, b, v0)
        clumping_label = rf'{A:.2g}\,(V_{{\rm vir}}/{v0:.0f})^{{{b:+.2g}}}'
    elif args.evolving_clumping:
        A, b, z0, fit_z, _fit_med, all_z, all_med, all_n = fit_evolving_clumping(
            z_all, cg_all, r_all, args.re_factor, args.height_factor, args.br06_height,
            args.clumping_fit_min_n, args.clumping_fit_z0)
        print(f'evolving clumping fit: clumping(z) = {A:.3g} * ((1+z)/{1.0 + z0:.1f})^{b:+.3g} '
              f'[fit to {fit_z.size}/{all_z.size} snapshots with >= {args.clumping_fit_min_n} '
              'galaxies]')
        for zv, med, nn in zip(all_z, all_med, all_n):
            used = 'fit' if nn >= args.clumping_fit_min_n else 'excluded (N too small)'
            print(f'  z={zv:5.2f}  N={nn:5d}  median required clumping={med:7.2f}  '
                  f'fitted={clumping_powerlaw(zv, A, b, z0):7.2f}  [{used}]')
        clumping_arg = clumping_powerlaw(z_all, A, b, z0)
        clumping_label = rf'{A:.2g}\,[(1{{+}}z)/{1.0 + z0:.0f}]^{{{b:+.2g}}}'
    else:
        clumping_arg = args.clumping
        clumping_label = None

    if not args.shell:
        n_all = cold_gas_density(cg_all, r_all, re_factor=args.re_factor,
                                 height_factor=args.height_factor, clumping=clumping_arg,
                                 br06_height=args.br06_height)
        config_label = density_config_label(args.re_factor, args.height_factor,
                                            args.br06_height, args.clumping, clumping_label)

    # Every available snapshot is plotted here (not a handful of representative
    # ones): both figures are now drawn from the identical gather_population()
    # sample, so a galaxy crossing n_fbk shows up as a point above the dashed
    # line here AND as a ringed point in the mass-redshift figure -- nothing in
    # one figure that the other's selection silently excluded.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = plt.Normalize(vmin=z_unique.min(), vmax=z_unique.max())

    for z_val in z_unique:
        sel = (z_all == z_val) & np.isfinite(n_all)
        ax.scatter(ms_all[sel], n_all[sel], s=10, alpha=0.5, color=CMAP(norm(z_val)),
                   edgecolors='none', zorder=2)

    ax.axhline(N_FBK_CM3, color='0.15', ls='--', lw=2.0, zorder=3)
    ax.text(0.02, N_FBK_CM3 * 1.2, r'Dekel+23 eq. 5: $n_{\rm fbk} = 2.23\times10^{3}\ \rm cm^{-3}$',
            transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=13)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, N_FBK_CM3 * 4.0))  # headroom for the threshold label
    ax.set_xlabel(r'Stellar Mass $(M_\odot)$')
    ax.set_ylabel(r'Shell number density, $n_{\rm sh}\ (\mathrm{cm}^{-3})$' if args.shell
                  else r'Cold-gas number density, $n\ (\mathrm{cm}^{-3})$')
    ax.set_title(config_label, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'Redshift, $z$')
    cbar.set_ticks(z_unique)
    cbar.set_ticklabels([f'{zv:.1f}' for zv in z_unique])

    outdir = args.outdir or os.path.join(args.dir, 'plots/')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'ShellDensity_FFBCriterion.pdf' if args.shell
                        else 'ColdGasDensity_FFBCriterion.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'wrote {path}')

    figure_mass_redshift_plane(
        z_all, logmv_all, n_all, outdir, config_label=config_label,
        density_label=(r'Shell number density, $n_{\rm sh}\ (\mathrm{cm}^{-3})$' if args.shell
                       else r'Cold-gas number density, $n\ (\mathrm{cm}^{-3})$'),
        filename=('MvirRedshiftPlane_ShellDensity.pdf' if args.shell
                  else 'MvirRedshiftPlane_ColdGasDensity.pdf'))

    ffb_regime_z = ffb_regime_logmv = ffb_regime = None
    if args.ffb_regime_dir is not None:
        ffb_hdr = header(args.ffb_regime_dir)
        (ffb_regime_z, _ffb_ms, ffb_regime_logmv, _ffb_n, _ffb_cg, _ffb_r, _ffb_vvir,
         ffb_regime, _ffb_rvir, _ffb_mdot) = gather_population(
            args.ffb_regime_dir, ffb_hdr, args.zmin, args.zmax, args.min_mstar,
            args.min_spin_percentile, args.min_particles)
        if ffb_regime_z.size == 0:
            raise SystemExit(f'No galaxies passed the selection cuts in {args.ffb_regime_dir} '
                              'for --ffb-regime-dir.')
        print(f'--ffb-regime-dir {args.ffb_regime_dir}: {ffb_regime_z.size} galaxies, '
              f'{np.count_nonzero(ffb_regime == 1)} with FFBRegime==1 '
              f'({100.0 * np.count_nonzero(ffb_regime == 1) / ffb_regime_z.size:.2f}%)')

    figure_ffb_criteria_comparison(
        z_all, logmv_all, n_all, outdir, args.comparison_zbins, args.comparison_mvbins,
        args.sigmoid_delta_dex, config_label,
        ffb_regime_z=ffb_regime_z, ffb_regime_logmv=ffb_regime_logmv,
        ffb_regime=ffb_regime, ffb_regime_dir=args.ffb_regime_dir)

    figure_density_mass_scatter(z_all, logmv_all, n_all, outdir, config_label)

    if args.shell:
        print('  skipping RequiredBoost figure: it is a disc-geometry diagnostic '
              '(required disc-radius shrinkage), which has no shell analogue')
    else:
        figure_required_boost(z_all, cg_all, r_all, outdir, args.height_factor,
                              args.br06_height)

    if args.grid:
        figure_clumping_law_grid(
            args.dir, hdr, args.zmin, args.zmax, args.min_mstar, outdir,
            args.sigmoid_delta_dex, args.comparison_zbins, args.comparison_mvbins,
            args.grid_fixed_clumping, args.clumping_fit_min_n, args.clumping_fit_z0,
            args.clumping_fit_v0, args.clumping_vvir_bins, args.grid_min_particles,
            args.grid_min_spin_percentile)


if __name__ == '__main__':
    main()

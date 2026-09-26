#!/usr/bin/env python
"""
Which galaxies each FFB criterion selects, in the (z, Mvir) plane.

Everything here reads FFBRegime straight out of the runs -- it is what SAGE26
itself flagged, not a criterion recomputed in post.  Six criteria are compared,
in a fixed panel order (see MODES):

    mode 1   Li+2024 sigmoid on halo mass
    mode 4   BK25 g_max acceleration, log-normal c
    mode 8   Dekel+2023 SHELL, n_sh > n_fbk (their eq. 62)
    mode 9   Dekel+2023 DISC, n_d > n_fbk AND Sigma_c > Sigma_crit (eqs. 64, 67)
    mode 10  shell AND disc-surface -- D23's own "robust" pairing of eqs. 62
             and 67.  NOTE this is not the set intersection of modes 8 and 9:
             it omits the disc's own 3D condition, so it is the wider set.
    mode 11  shell OR disc -- the union of modes 8 and 9, i.e. either geometry
             suffices, which is how D23 frame them (two limiting scenarios
             selected by the streams' angular momentum, not two tests every
             galaxy must pass).

The Dekel+23 eq. 62 / Li+24 eq. 2 threshold curve is drawn on every panel.  A
mode whose run is missing from output/ is skipped with a message rather than
crashing the whole figure set.

    FFBSelection_Actual.pdf              both simulations overlaid as a
                                         scatter, coloured by simulation

    FFBSelection_Density.pdf             the same comparison as a number-density
                                         map of the selected galaxies: one row
                                         per simulation, one column per
                                         criterion, on a single shared log
                                         colour scale, so a criterion can be
                                         compared across boxes by reading down
                                         a column.  Cells whose selections are
                                         mostly haloes with an untrustworthy
                                         disc size are greyed rather than
                                         dropped -- see read_selection()'s
                                         reliable flag.

    FFBThresholds_MvirRedshift.pdf       the threshold relations on one
                                         (z, Mvir) plane, with the fitted
                                         exponent x in M_v,ffb ~ (1+z)^-x in
                                         the legend.  Colour is the criterion,
                                         line style the simulation.  A criterion
                                         with no 50%-selection mass appears as a
                                         16-84% band instead of a line, and one
                                         with too few selections to characterise
                                         appears as open markers -- which of
                                         them do is read off the data, not
                                         hard-coded.  Mode 9 falls back because
                                         what decides it is the halo's spin,
                                         which is scatter within a mass bin
                                         rather than a function of mass.

    FFBFraction_vs_Redshift.pdf          the selected fraction against redshift,
                                         one panel per simulation and one colour
                                         per criterion -- the transpose of the
                                         figures above, where colour is the
                                         simulation and the panel is the
                                         criterion.  Its palette is therefore a
                                         separate one; see MODE_STYLE.

The runs of a simulation share a halo catalogue, so their galaxy populations
are identical (verified: 737594 galaxies at 5<z<14 in each microuchuu run) --
only the FFBRegime flag differs between them.

Why the two criteria are not expected to agree
----------------------------------------------
Mode 1 IS the eq. 62 curve, plus a 0.15 dex sigmoid, so its panel necessarily
straddles the line -- that panel is a consistency check, not a result.  Its
apparent spread comes from that sigmoid convolved with a steep halo mass
function: at 0.5 dex below the threshold the selection probability is only
~4%, but there are enough haloes there that the bin still contributes ~20% of
all mode-1 selections.  Narrowing the sigmoid to 0.01 dex collapses the spread
from 1.5 dex below the line to 0.04 dex.

Mode 8 is a forward calculation from SAGE26's own accretion rate through
Dekel+23's eqs. 38-41, with no width parameter at all; its spread is real
halo-to-halo scatter in Mdot_ac (~0.29 dex).  Where it departs from the line
is where SAGE26's accretion history departs from the analytic eq. 31 that
curve assumes.

    python plotting/ffb_selection_comparison.py
    python plotting/ffb_selection_comparison.py --zmin 5 --zmax 14
    # a different machine: different staging directory, box names and naming
    python plotting/ffb_selection_comparison.py \\
        --output-root output/testmodels --naming hpc \\
        --sim-dirs Millennium miniUchuu
    python plotting/ffb_selection_comparison.py --dilute 40000   # thin the grey background

SAGE26 -- released under MIT (see LICENSE).
"""

import argparse
import glob
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch

# ===========================================================================
# Self-contained copies of the few helpers this script needs.
#
# Vendored on purpose: this file is routinely copied on its own to a cluster
# where the rest of plotting/ is not present, and an ImportError there is a
# worse failure than a little duplication here.  These are byte-for-byte the
# definitions in ffb_cold_gas_density.py -- if that file's versions change,
# change these too.
# ===========================================================================

MSUN_G = 1.989e33          # solar mass [g]
PROTON_MASS_G = 1.6726e-24 # proton mass [g]
MU_NEUTRAL = 1.2           # Dekel+23 sec. 2: mean molecular weight, neutral H+He, T < 10^4 K
KM_TO_CM = 1.0e5
GYR_TO_S = 3.1557e16

# Dekel et al. (2023) eq. (5): feedback-free density threshold.
N_FBK_CM3 = 2.23e3

# Dekel et al. (2023) eq. (62) / Li et al. (2024) eq. (2): the FFB threshold in
# the halo-mass/redshift plane, M_v,ffb(z) = 10^10.8 Msun * ((1+z)/10)^-6.2.
FFB_THRESHOLD_NORM_LOG_MSUN = 10.8
FFB_THRESHOLD_SLOPE = -6.2

# Bullock spin parameter, exactly as get_disk_radius() computes it in
# model_misc.c: lambda' = |Spin| / (SQRT_REPLACEMENT * Vvir * Rvir).  1.414
# (not sqrt(2)) is intentional -- see SQRT_REPLACEMENT in model_misc.c.
SQRT_REPLACEMENT = 1.414

# Dekel+23 shell scenario (eqs. 38-41), used for mode 8's threshold curve.
STREAM_RADIUS_FRACTION = 0.05   # R_str / Rvir, D23's fiducial (their eq. 61)
SHELL_SOUND_SPEED_KMS = 13.0    # post-shock c_s at T ~ 10^4 K -> Mach ~ 15 at Vvir ~ 200


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


def spin_parameter(spinx, spiny, spinz, vvir, rvir):
    """Bullock spin parameter lambda', matching get_disk_radius() exactly."""
    spin_mag = np.sqrt(spinx**2 + spiny**2 + spinz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return spin_mag / (SQRT_REPLACEMENT * vvir * rvir)


# FIRE (Muratov et al. 2015) critical circular velocity separating the two
# reheating power-law slopes, exactly matching FIRE_V_CRIT_KMS in
# model_starformation_and_feedback.c.


def ffb_threshold_mass_msun(z):
    """Dekel+23 eq. (62): FFB threshold halo mass [Msun] at redshift z."""
    return 10.0 ** (FFB_THRESHOLD_NORM_LOG_MSUN
                     + FFB_THRESHOLD_SLOPE * np.log10((1.0 + z) / 10.0))

# Width (in dex of Mvir) of the smooth sigmoid straddling M_v,ffb(z), Li et
# al. (2024) eq. (3) -- exactly what calculate_ffb_fraction() in
# model_regimes.c uses for FeedbackFreeModeOn in {1, 6}. A halo-mass-only
# criterion, with no dependence on any simulated gas structure.


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


def read_shell_inputs(directory, zmin, zmax):
    """(z, log10 Mvir, Vvir, Rvir_cm, Mdot_ac) for every central in a run.

    A lean stand-in for gather_population(): mode 8's threshold curve is the
    only thing here that needs the no-FFB run, and it needs exactly these five
    arrays, so there is no reason to carry the disc-geometry machinery.

    Mdot_ac = BaryonFrac * dMvir/dt across consecutive snapshots, matched by
    GalaxyIndex (stable in SAGE26's output).  NaN where a galaxy has no
    progenitor or the halo lost mass.
    """
    hdr = header(directory)
    snaps = [s for s in hdr['output_snaps'] if zmin <= hdr['z'][s] <= zmax]

    def snapshot(snap):
        gid_l, mv_l, rv_l, vv_l = [], [], [], []
        for path in model_files(directory):
            with h5py.File(path, 'r') as f:
                grp = f.get(f'Snap_{snap}')
                if grp is None:
                    continue
                mv = np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun']
                keep = (mv > 0) & (np.asarray(grp['Type'], dtype=np.int32) == 0)
                gid_l.append(np.asarray(grp['GalaxyIndex'], dtype=np.int64)[keep])
                mv_l.append(mv[keep])
                rv_l.append(np.asarray(grp['Rvir'], dtype=np.float64)[keep] * hdr['to_cm'])
                vv_l.append(np.asarray(grp['Vvir'], dtype=np.float64)[keep])
        if not gid_l or sum(a.size for a in gid_l) == 0:
            return None
        return tuple(np.concatenate(a) for a in (gid_l, mv_l, rv_l, vv_l))

    def all_mvir(snap):
        """GalaxyIndex -> Mvir for a whole snapshot, sorted, unfiltered."""
        gid_l, mv_l = [], []
        for path in model_files(directory):
            with h5py.File(path, 'r') as f:
                grp = f.get(f'Snap_{snap}')
                if grp is None:
                    continue
                gid_l.append(np.asarray(grp['GalaxyIndex'], dtype=np.int64))
                mv_l.append(np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun'])
        if not gid_l:
            return np.array([], dtype=np.int64), np.array([])
        gid, mv = np.concatenate(gid_l), np.concatenate(mv_l)
        order = np.argsort(gid)
        return gid[order], mv[order]

    z_l, lm_l, vv_l, rv_l, md_l = [], [], [], [], []
    for snap in snaps:
        cur = snapshot(snap)
        if cur is None:
            continue
        gid, mv, rvir, vvir = cur

        mdot = np.full(mv.size, np.nan)
        if snap - 1 >= 0:
            pg, pm = all_mvir(snap - 1)
            if pg.size:
                idx = np.clip(np.searchsorted(pg, gid), 0, pg.size - 1)
                prev = np.where(pg[idx] == gid, pm[idx], np.nan)
                dt = (cosmic_time_gyr(hdr['z'][snap], hdr['omega_m'], hdr['omega_l'],
                                      hdr['hubble_h'])
                      - cosmic_time_gyr(hdr['z'][snap - 1], hdr['omega_m'],
                                        hdr['omega_l'], hdr['hubble_h']))
                if dt > 0:
                    mdot = hdr['baryon_frac'] * (mv - prev) / dt

        z_l.append(np.full(mv.size, hdr['z'][snap]))
        lm_l.append(np.log10(mv))
        vv_l.append(vvir)
        rv_l.append(rvir)
        md_l.append(mdot)

    if not z_l:
        raise SystemExit(f'No centrals in {directory} for {zmin} <= z <= {zmax}.')
    return tuple(np.concatenate(a) for a in (z_l, lm_l, vv_l, rv_l, md_l))


HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'kieren_cohare_palatino_sty.mplstyle')

# One entry per simulation, in fixed order -- the colour follows the
# simulation, never its position in a panel, so a galaxy's colour means the
# same thing in both panels. Blue/orange is the canonical dichromat-safe pair;
# both were checked against each other and against the eq. 62 curve's red
# under deuteranopia/protanopia/tritanopia (OKLab dE: 32.9 normal, >=26.6 CVD
# between the two; 11.9 deuteranopic against the red curve) and for contrast
# on white (7.9:1 and 5.2:1).
# Each simulation is a base run-directory name plus a colour.  Every criterion
# lives in a sibling directory formed as base + MODE_DIR_SUFFIX, so moving to a
# different machine or a differently-named box is one --sim-dirs argument
# rather than a dozen edits.  The colour follows the simulation, never its
# position in a panel.  Blue/orange is the canonical dichromat-safe pair; both
# were checked against each other and against the eq. 62 curve's red under
# deuteranopia/protanopia/tritanopia (OKLab dE: 32.9 normal, >=26.6 CVD between
# the two; 11.9 deuteranopic against the red curve) and for contrast on white
# (7.9:1 and 5.2:1).
SIM_COLOURS = ['#1F45C0', '#A35A00']
DEFAULT_SIM_DIRS = ['millennium', 'microuchuu']

# Directory suffix per criterion, appended to a simulation's base name.  The
# naming differs between machines, so it is a named scheme rather than a
# constant: pick with --naming, override any single entry with --mode-dir.
# The '_noffb' key is the no-FFB run, used only for mode 8's threshold curve.
NAMING_SCHEMES = {
    'local': {
        'mode1': '', 'mbk': '_mbk_smooth', 'mode8': '_ffbmode8',
        'mode9': '_ffbmode9', 'mode10': '_ffbmode10', 'mode11': '_ffbmode11',
        '_noffb': '_noffb',
    },
    'hpc': {
        'mode1': '_FFB', 'mbk': '_FFB_MBK25', 'mode8': '_ffbmode8',
        'mode9': '_ffbmode9', 'mode10': '_ffbmode10', 'mode11': '_ffbmode11',
        '_noffb': '_noFFB',
    },
}

# How a base directory name is shown in the figures.  Anything not listed is
# displayed as-is, so a new box needs no entry.
# Only names whose casing cannot be inferred need an entry; everything else is
# shown exactly as the directory is named, so "Millennium" stays capitalised.
SIM_DISPLAY_NAME = {'microuchuu': 'microUchuu', 'miniuchuu': 'miniUchuu'}


def build_sims(sim_dirs, output_root, naming='local', overrides=None):
    """[(display label, colour, {mode key: full run path}), ...].

    overrides is {key: suffix}, applied on top of the chosen naming scheme, so
    one oddly-named run does not need a new scheme.
    """
    suffixes = dict(NAMING_SCHEMES[naming])
    suffixes.update(overrides or {})
    sims = []
    for i, base in enumerate(sim_dirs):
        label = SIM_DISPLAY_NAME.get(base.lower(), base)
        dirs = {key: os.path.join(output_root, base + suffix)
                for key, suffix in suffixes.items()}
        sims.append((label, SIM_COLOURS[i % len(SIM_COLOURS)], dirs))
    return sims


# Populated by main() from --sim-dirs / --output-root.
SIMS = []

# The criteria compared, in panel order. Each is a separate run of the same
# halo catalogue, so the galaxy populations are identical and only FFBRegime
# differs -- verified above.
MODES = [
    ('mode1', r'FeedbackFreeModeOn=1' '\n' r'Li+24 sigmoid'),
    ('mbk', r'FeedbackFreeModeOn=4' '\n' r'BK25 $g_{\rm max}$, log-normal $c$'),
    ('mode8', r'FeedbackFreeModeOn=8' '\n' r'Dekel+23 shell $n_{\rm sh}$'),
    ('mode9', r'FeedbackFreeModeOn=9' '\n' r'Dekel+23 disc $n_d$ + $\Sigma_c$'),
    ('mode10', r'FeedbackFreeModeOn=10' '\n' r'D+23 AND: eq.62 $\wedge$ eq.67'),
    ('mode11', r'FeedbackFreeModeOn=11' '\n' r'D+23 OR: shell $\vee$ disc'),
]

# (label, colour, marker) per criterion, for the figures where colour encodes
# the CRITERION rather than the simulation.  NOTE this palette deliberately
# does NOT share meaning with SIMS above -- there colour is the simulation;
# here colour is the criterion and the simulation is the panel or line style.
#
# Six categories is past where colour alone is reliable, so every series also
# carries a distinct MARKER -- identity is never colour-alone.  The hues were
# solved rather than picked: modes 1, 4 and 8 were held fixed (they have been
# read across several figures already) and the other three searched against
# them, enforcing the normal-vision floor of dE 15 as a hard constraint that
# secondary encoding does NOT excuse.  Result: worst-pair OKLab dE 18.8 with
# normal vision, 11.1 under deuteranopia/protanopia/tritanopia, minimum
# contrast 3.3:1 on white.
#  Five categories is past the point where colour
# alone is reliable under colour-vision deficiency -- the best available
# 5-hue set against this one's constraints reaches only dE 13.1 under CVD and
# 16.7 with normal vision, with the two Dekel+23 disc-involving modes as the
# binding pair -- so every series also carries a distinct MARKER. Identity is
# never colour-alone here.
MODE_STYLE = {
    'mode1': (r'mode 1: Li+24 sigmoid', '#1F45C0', 'o'),
    'mbk': (r'mode 4: BK25 $g_{\rm max}$', '#117733', 's'),
    'mode8': (r'mode 8: Dekel+23 shell', '#A35A00', '^'),
    'mode9': (r'mode 9: Dekel+23 disc', '#932699', 'D'),
    'mode10': (r'mode 10: shell AND disc', '#CC70A7', 'v'),
    'mode11': (r'mode 11: shell OR disc', '#0B074C', 'P'),
}

BACKGROUND_GREY = '#C8C8C8'   # unselected galaxies: neutral, never a series colour
THRESHOLD_RED = '#D62728'     # eq. 62 / Li+24 eq. 2 curve on the scatter figure
THRESHOLD_CYAN = '#00E5E5'    # the same curve over plasma, which is red at its top end
INK = '#222222'

SEED = 20260926
DILUTE_BACKGROUND = None     # cap on grey points per simulation per panel
MV_BIN_DEX = 0.15            # log10(Mvir) bin height for the density maps (--mv-bin-dex)

# A halo's disc size is only as trustworthy as its spin vector, and
# get_disk_radius() applies no floor, so a shot-noise spin becomes an
# arbitrarily small disc.  Cells on the density maps whose selections are
# mostly haloes failing these are greyed out rather than dropped.
RESOLVED_MIN_PARTICLES = 20
RESOLVED_SPIN_PERCENTILE = 1.0
# A cell is greyed when fewer than this fraction of its selections are reliable.
RELIABLE_CELL_FRACTION = 0.25
UNRELIABLE_GREY = '#9A9A9A'




def read_selection(directory, hdr, zmin, zmax, centrals_only=True):
    """Every galaxy in [zmin, zmax]: redshift, log10(Mvir/Msun) and FFBRegime.

    Deliberately NOT gather_population(): that applies ColdGas > 0 and
    DiskRadius > 0 cuts, which are right for a disc-density figure and wrong
    here -- the mode-8 shell criterion uses no ColdGas at all, so a
    cold-gas-free galaxy can legitimately be FFBRegime==1 and must not be
    filtered out before we count it.

    centrals_only=True (the default) restricts every criterion to Type == 0,
    so all three are evaluated on exactly the same galaxies. This is needed
    for a fair comparison because mode 8 is centrals-only in the model itself
    (model_regimes.c): its accretion rate is infall_recipe()'s FoF-group-wide
    infallingGas, credited to the central, and a satellite inside a larger
    halo has no cosmic-web stream of its own to shock -- it is being stripped.
    Modes 1 and 4 have no such restriction and do flag a handful of
    satellites (9 of 730 in millennium, 17 of 1553 in microuchuu, over
    5<z<14); masking them here removes that asymmetry rather than inventing a
    per-satellite accretion rate that SAGE26 does not compute. Satellites are
    1.5-2.3% of the population at these redshifts and there are no Type 2
    orphans at all, so the figures barely move -- the point is that the
    denominator is then identical across criteria.

Also returns 'reliable', a per-galaxy flag marking haloes whose disc size
    can be trusted: at least RESOLVED_MIN_PARTICLES N-body particles AND above
    the RESOLVED_SPIN_PERCENTILE'th percentile of the spin distribution.
    Nothing is filtered on it -- the density maps grey out the cells it
    condemns instead, so the affected region is visible rather than silently
    deleted.

    The two conditions select nearly the same haloes and are applied together
    for that reason: get_disk_radius() in model_misc.c sets the disc radius
    straight from the halo's spin vector with no floor, so a poorly-resolved
    halo -- whose spin vector is shot-noise-dominated -- gets an arbitrarily
    small disc and an arbitrarily large density.  That matters far more for the
    Dekel+23 DISC criterion than for anything else here, since its 3D condition
    goes as lambda^-13/6 and so selects almost entirely on spin.
    """
    snaps = [s for s in hdr['output_snaps'] if zmin <= hdr['z'][s] <= zmax]
    z_l, logmv_l, ffb_l, spin_l, len_l_all = [], [], [], [], []
    for snap in snaps:
        mv_l, f_l, sp_l, len_l = [], [], [], []
        for path in model_files(directory):
            with h5py.File(path, 'r') as f:
                grp = f.get(f'Snap_{snap}')
                if grp is None:
                    continue
                mv = np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun']
                keep = mv > 0
                if centrals_only:
                    keep &= np.asarray(grp['Type'], dtype=np.int32) == 0
                mv_l.append(mv[keep])
                f_l.append(np.asarray(grp['FFBRegime'], dtype=np.int32)[keep])
                len_l.append(np.asarray(grp['Len'], dtype=np.int64)[keep])
                sp_l.append(spin_parameter(
                    np.asarray(grp['Spinx'], dtype=np.float64)[keep],
                    np.asarray(grp['Spiny'], dtype=np.float64)[keep],
                    np.asarray(grp['Spinz'], dtype=np.float64)[keep],
                    np.asarray(grp['Vvir'], dtype=np.float64)[keep],
                    np.asarray(grp['Rvir'], dtype=np.float64)[keep]))
        if not mv_l or sum(a.size for a in mv_l) == 0:
            continue
        mv = np.concatenate(mv_l)
        z_l.append(np.full(mv.size, hdr['z'][snap]))
        logmv_l.append(np.log10(mv))
        ffb_l.append(np.concatenate(f_l))
        spin_l.append(np.concatenate(sp_l))
        len_l_all.append(np.concatenate(len_l))
    if not z_l:
        raise SystemExit(f'No galaxies in {directory} for {zmin} <= z <= {zmax}.')

    pop = {'z': np.concatenate(z_l), 'logmv': np.concatenate(logmv_l),
           'ffb': np.concatenate(ffb_l), 'spin': np.concatenate(spin_l),
           'len': np.concatenate(len_l_all)}

    finite = np.isfinite(pop['spin'])
    floor = (np.percentile(pop['spin'][finite], RESOLVED_SPIN_PERCENTILE)
             if np.any(finite) else 0.0)
    pop['reliable'] = (finite & (pop['spin'] >= floor)
                       & (pop['len'] >= RESOLVED_MIN_PARTICLES))
    return pop


def threshold_curve(ax, zlim, colour):
    z_curve = np.linspace(zlim[0], zlim[1], 300)
    ax.plot(z_curve, np.log10(ffb_threshold_mass_msun(z_curve)), color=colour,
            ls='--', lw=2.2, zorder=5,
            label=r'$M_{\rm v,ffb}(z)$ (eq. 62 / Li+24 eq. 2)')


def snapshot_z_edges(z_values):
    """Bin edges midway between consecutive snapshot redshifts.

    The output redshifts are discrete, so a uniform histogram in z would put
    empty stripes between populated snapshots and make the map look like it
    has structure it does not. One column per snapshot instead.
    """
    zs = np.unique(z_values)
    mids = 0.5 * (zs[:-1] + zs[1:])
    first = zs[0] - (mids[0] - zs[0]) if zs.size > 1 else zs[0] - 0.1
    last = zs[-1] + (zs[-1] - mids[-1]) if zs.size > 1 else zs[-1] + 0.1
    return np.concatenate([[first], mids, [last]])


# ============================ scatter figure ============================

def draw_scatter_panel(ax, entries, zlim, mvlim, title, dilute, rng):
    for _label, _colour, pop, selected in entries:
        unsel = ~selected
        if dilute and np.count_nonzero(unsel) > dilute:
            idx = np.flatnonzero(unsel)
            unsel = np.zeros_like(selected)
            unsel[rng.choice(idx, dilute, replace=False)] = True
        ax.scatter(pop['z'][unsel], pop['logmv'][unsel], s=2, alpha=0.18,
                   color=BACKGROUND_GREY, edgecolors='none', zorder=1, rasterized=True)

    for label, colour, pop, selected in entries:
        ax.scatter(pop['z'][selected], pop['logmv'][selected], s=9, alpha=0.65,
                   color=colour, edgecolors='none', zorder=3, rasterized=True,
                   label=f'{label} ({np.count_nonzero(selected)})')

    threshold_curve(ax, zlim, THRESHOLD_RED)
    ax.set_xlim(*zlim)
    ax.set_ylim(*mvlim)
    ax.set_xlabel(r'Redshift, $z$')
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)


def make_scatter_figure(panels, outdir, zlim, mvlim, dilute):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.0),
                             sharex=True, sharey=True)
    fig.set_tight_layout(False)

    for ax, (title, entries) in zip(np.atleast_1d(axes), panels):
        draw_scatter_panel(ax, entries, zlim, mvlim, title, dilute, rng)

    axes[0].set_ylabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$')

    # One legend for the figure: identity is never colour-alone, and the
    # counts sit in the label rather than as numbers on the points.
    handles = [Line2D([], [], marker='o', ls='none', ms=7, color=colour, label=label)
               for label, colour, _, _ in panels[0][1]]
    handles.append(Line2D([], [], color=BACKGROUND_GREY, marker='o', ls='none', ms=6,
                          label='not selected'))
    handles.append(Line2D([], [], color=THRESHOLD_RED, ls='--', lw=2.2,
                          label=r'$M_{\rm v,ffb}(z)$ (eq. 62 / Li+24 eq. 2)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Galaxies SAGE26 itself flagged as FFB (FFBRegime = 1), from each run',
                 fontsize=12.5, color=INK, y=0.99)
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.06)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBSelection_Actual.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}')


# ============================ density figures ===========================

def make_density_figure(by_sim, modes_present, outdir, zlim, mvlim, mv_bin_dex):
    """Number density of the SELECTED galaxies in (z, Mvir): one row per
    simulation, one column per criterion, on a single shared log colour scale.

    One figure rather than one per simulation, so a criterion can be compared
    across boxes by reading down a column.  The scale is shared across all
    panels; the two simulations' selection counts are within a factor of a few,
    so neither row washes out.

    Bins one column per output snapshot (see snapshot_z_edges()) -- the finest
    honest binning in z, since every galaxy in a snapshot sits at exactly the
    same redshift.

    Cells whose selections are mostly haloes with an untrustworthy disc size
    are drawn in GREY rather than on the colour scale, and are not dropped.
    "Untrustworthy" is read_selection()'s reliable flag: fewer than
    RESOLVED_MIN_PARTICLES particles, or in the bottom RESOLVED_SPIN_PERCENTILE
    per cent of the spin distribution.  Greying rather than cutting keeps the
    affected region visible, which matters because it is not uniformly spread
    -- it concentrates exactly where the Dekel+23 disc criterion does its
    selecting, its 3D condition going as lambda^-13/6.
    """
    sims = [(label, by_sim[label]) for label, _c, _d in SIMS if by_sim.get(label)]
    if not sims:
        return
    # Redshift edges PER SIMULATION, not shared.  The two output at different
    # redshifts -- 12 snapshots for millennium, 11 for microUchuu, only one of
    # which coincides -- so a shared edge set would make 23 columns that each
    # simulation fills only half of, reintroducing exactly the empty stripes
    # snapshot_z_edges() exists to avoid.  The mass edges ARE shared, since
    # both rows bin the same quantity the same way.
    z_edges_for = {label: snapshot_z_edges(
                       np.concatenate([p['z'] for p, _s in modes.values()]))
                   for label, modes in sims}
    mv_edges = np.arange(mvlim[0], mvlim[1] + mv_bin_dex, mv_bin_dex)

    def cell_grids(pop, sel, z_edges):
        total, _, _ = np.histogram2d(pop['z'][sel], pop['logmv'][sel],
                                     bins=[z_edges, mv_edges])
        good = sel & pop['reliable']
        trusted, _, _ = np.histogram2d(pop['z'][good], pop['logmv'][good],
                                       bins=[z_edges, mv_edges])
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(total > 0, trusted / total, np.nan)
        suspect = (total > 0) & (frac < RELIABLE_CELL_FRACTION)
        return (np.ma.masked_where((total == 0) | suspect, total),
                np.ma.masked_where(~suspect, total))

    panels = {}
    for label, modes in sims:
        for key, _title in modes_present:
            if key in modes:
                panels[(label, key)] = cell_grids(*modes[key], z_edges_for[label])

    vmax = max((g.max() for g, _ in panels.values() if g.count()), default=1)
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('white')
    grey_cmap = ListedColormap([UNRELIABLE_GREY])

    ncols = len(modes_present)
    # Per-panel geometry deliberately matches the single-row version (6.4 x 6.0):
    # shrinking panels to make two rows "fit" squashes the cells, and the cell
    # size is the whole point of this figure.  A wide figure is fine in a PDF.
    fig, axes = plt.subplots(len(sims), ncols, figsize=(6.4 * ncols, 6.0 * len(sims)),
                             sharex=True, sharey=True, squeeze=False)
    fig.set_tight_layout(False)

    n_suspect = 0
    mesh = None
    for row, (label, modes) in enumerate(sims):
        for col, (key, title) in enumerate(modes_present):
            ax = axes[row][col]
            if (label, key) not in panels:
                ax.set_visible(False)
                continue
            grid, grey = panels[(label, key)]
            z_edges = z_edges_for[label]
            mesh = ax.pcolormesh(z_edges, mv_edges, grid.T, cmap=cmap, norm=norm,
                                 shading='flat', zorder=2)
            if grey.count():
                n_suspect += int(grey.count())
                ax.pcolormesh(z_edges, mv_edges, grey.T, cmap=grey_cmap,
                              shading='flat', zorder=2)
            # Cyan, not the scatter figure's red: plasma runs through red/orange
            # at its high end, where a red curve would vanish into dense cells.
            threshold_curve(ax, zlim, THRESHOLD_CYAN)
            ax.set_xlim(*zlim)
            ax.set_ylim(*mvlim)
            _pop, sel = modes[key]
            if row == 0:
                ax.set_title(f'{title}\n{np.count_nonzero(sel)} selected',
                             fontsize=10.5, color=INK, pad=8)
            else:
                ax.set_title(f'{np.count_nonzero(sel)} selected', fontsize=10.5,
                             color=INK, pad=8)
            if row == len(sims) - 1:
                ax.set_xlabel(r'Redshift, $z$')
            if col == 0:
                ax.set_ylabel(f'{label}\n' r'$\log_{10}(M_{\rm vir}/M_\odot)$',
                              fontsize=12)

    handles = [Line2D([], [], color=THRESHOLD_CYAN, ls='--', lw=2.2,
                      label=r'$M_{\rm v,ffb}(z)$ (eq. 62 / Li+24 eq. 2)')]
    if n_suspect:
        handles.append(Patch(facecolor=UNRELIABLE_GREY,
                             label=f'mostly unresolved: Len $<$ {RESOLVED_MIN_PARTICLES} '
                                   f'or spin below the {RESOLVED_SPIN_PERCENTILE:g}th pct'))
    axes[-1][-1].legend(handles=handles, loc='lower left', frameon=True,
                        framealpha=0.85, fontsize=8)

    # subplots_adjust BEFORE the colorbar: fig.colorbar(ax=...) shrinks the
    # axes to make room for itself, and a later subplots_adjust would undo
    # that shrink and leave the bar sitting on top of the last panel.
    fig.subplots_adjust(top=0.90, wspace=0.06, hspace=0.10)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.015, pad=0.012)
    cbar.set_label('FFB-selected galaxies per bin')

    fig.suptitle(r'FFB selection density, bins of one snapshot $\times$ '
                 f'{mv_bin_dex:g} dex', fontsize=14, color=INK, y=0.99)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBSelection_Density.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}   ({n_suspect} cells greyed as unresolved)')


def make_fraction_figure(by_sim, outdir, zlim, population_label='central galaxies'):
    """FFB-selected fraction against redshift, one panel per simulation and
    one colour per criterion -- the transpose of the other figures here, where
    colour is the simulation and the panel is the criterion.

    The denominator is whatever read_selection() returned -- by default the
    CENTRAL galaxies of the snapshot, so all three criteria share exactly the
    same denominator (see read_selection()'s centrals_only). It is also
    exactly shared between the runs, since the three runs of a simulation
    have identical populations.

    Bands are Poisson (sqrt(N_selected) / N_total): at the high-redshift end a
    snapshot can hold only a handful of galaxies, and without them a fraction
    of 1/1 reads as indistinguishable from a well-measured 100%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    fig.set_tight_layout(False)

    for ax, (label, _colour, _dirs) in zip(axes, SIMS):
        for key, _title in MODES:
            if key not in by_sim[label]:
                continue
            pop, sel = by_sim[label][key]
            short, colour, mk = MODE_STYLE[key]
            zs = np.unique(pop['z'])
            n_tot = np.array([np.count_nonzero(pop['z'] == z) for z in zs], dtype=float)
            n_sel = np.array([np.count_nonzero(sel & (pop['z'] == z)) for z in zs],
                             dtype=float)
            frac = n_sel / n_tot
            err = np.sqrt(n_sel) / n_tot

            drawn = frac > 0
            ax.plot(zs[drawn], frac[drawn], color=colour, lw=1.8, marker=mk, ms=4.5,
                    zorder=3, label=short)
            ax.fill_between(zs[drawn], np.clip(frac - err, 1e-6, None)[drawn],
                            (frac + err)[drawn], color=colour, alpha=0.18,
                            lw=0, zorder=2)

        ax.set_yscale('log')
        ax.set_xlim(*zlim)
        ax.set_xlabel(r'Redshift, $z$')
        ax.set_title(label, fontsize=12, color=INK, pad=8)

    axes[0].set_ylabel(f'FFB fraction of {population_label}')
    axes[0].legend(loc='lower right', frameon=False, fontsize=9.5)

    fig.suptitle('Fraction of galaxies flagged FFB (FFBRegime = 1) against redshift',
                 fontsize=12.5, color=INK, y=0.99)
    fig.subplots_adjust(top=0.84, wspace=0.05)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBFraction_vs_Redshift.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}')


def threshold_from_selection(logmv, flagged, step=0.1, nmin=10):
    """log10 of the halo mass at which 50% of haloes satisfy a criterion.

    Works for the mass-threshold criteria (modes 1 and 4), whose selection
    probability does cross 0.5 at a well-defined mass. Returns NaN when it
    never does -- which is the normal case for mode 8 below z ~ 9, hence
    threshold_from_shell_density() for that one.
    """
    if flagged.sum() < 5:
        return np.nan
    edges = np.arange(np.floor(logmv.min() * 10) / 10, logmv.max() + step, step)
    idx = np.digitize(logmv, edges) - 1
    frac = np.array([flagged[idx == k].mean() if (idx == k).sum() >= nmin else np.nan
                     for k in range(len(edges) - 1)])
    hit = np.where(np.isfinite(frac) & (frac >= 0.5))[0]
    return 0.5 * (edges[hit[0]] + edges[hit[0] + 1]) if hit.size else np.nan


def threshold_from_shell_density(logmv, n_sh, step=0.15, nmin=15):
    """log10 of the halo mass where the MEDIAN shell density reaches n_fbk.

    Mode 8's 50%-selection mass is undefined below z ~ 9 -- no halo in the box
    is dense enough -- so the threshold is instead read off the median
    n_sh(Mvir) relation and solved for n_sh = n_fbk. Where that lands above
    the most massive halo present it is an extrapolation of the relation, not
    a statement that such haloes exist; the caller is told which points those
    are so they can be drawn open.
    """
    ok = np.isfinite(n_sh) & (n_sh > 0)
    if ok.sum() < 100:
        return np.nan, False
    lm, ln = logmv[ok], np.log10(n_sh[ok])
    edges = np.arange(10.0, 13.2, step)
    idx = np.digitize(lm, edges) - 1
    cx, cy = [], []
    for k in range(len(edges) - 1):
        s = idx == k
        if s.sum() >= nmin:
            cx.append(0.5 * (edges[k] + edges[k + 1]))
            cy.append(np.median(ln[s]))
    if len(cx) < 4:
        return np.nan, False
    slope, inter = np.polyfit(cx, cy, 1)
    m_thr = (np.log10(N_FBK_CM3) - inter) / slope
    return m_thr, bool(m_thr > lm.max())


def make_threshold_figure(outdir, zmin, zmax):
    """All three FFB thresholds together in the (z, Mvir) plane.

    Colour is the criterion (MODE_STYLE); line style is the simulation, so
    identity is never carried by colour alone. Points are measured per
    snapshot, lines are power-law fits log10 M_thr = A + slope*log10(1+z),
    and the slope of each is printed in the legend -- that slope is the
    exponent x in M_v,ffb ~ (1+z)^-x.

    The estimator differs by necessity between criteria and is stated in the
    legend: modes 1 and 4 use the 50%-selection mass from their own runs,
    mode 8 the mass at which the median n_sh reaches n_fbk (its 50%-selection
    mass does not exist below z ~ 9). Open markers on the mode-8 points mark
    redshifts where that mass exceeds the most massive halo in the box, i.e.
    the relation extrapolated beyond where it can be tested.
    """
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    fig.set_tight_layout(False)
    # Keyed by mode so the legend can be emitted in MODES order regardless of
    # the order the curves happen to be computed in (mode 8 is measured last,
    # by a different estimator, but belongs third).
    handles = {}

    # Only the first simulation.  With two, this figure carried 2 x 6 curves
    # and a 13-entry legend that was unreadable; one simulation keeps it to
    # one curve per criterion, which is what the figure is actually for.
    for sim_i, (label, _colour, dirs) in enumerate(SIMS[:1]):
        ls = '-' if sim_i == 0 else '--'
        marker = 'o' if sim_i == 0 else 's'

        # Every criterion that is read straight off its own run: try for the
        # 50%-selection mass, and fall back to showing where the selections
        # actually live if no such mass exists. Which modes fall back is a
        # property of the data, not hard-coded -- mode 9 has no threshold
        # because what decides it is the halo's spin, which is scatter WITHIN
        # a mass bin rather than a function of mass, and mode 10 may select
        # too few galaxies to define one at all.
        for key in ('mode1', 'mbk', 'mode9', 'mode10', 'mode11'):
            rundir = dirs[key]
            if not glob.glob(os.path.join(rundir, 'model_*.hdf5')):
                continue
            pop = read_selection(rundir, header(rundir), zmin, zmax)
            short, colour, _mk = MODE_STYLE[key]

            zs, ms_thr = [], []
            for zv in np.unique(pop['z']):
                m = pop['z'] == zv
                t = threshold_from_selection(pop['logmv'][m], pop['ffb'][m] == 1)
                if np.isfinite(t):
                    zs.append(zv); ms_thr.append(t)

            if len(zs) >= 3:
                zs, ms_thr = np.array(zs), np.array(ms_thr)
                slope, inter = np.polyfit(np.log10(1 + zs), ms_thr, 1)
                ax.plot(zs, ms_thr, marker, color=colour, ms=5, alpha=0.75, zorder=3)
                zf = np.linspace(zs.min(), zs.max(), 100)
                ax.plot(zf, inter + slope * np.log10(1 + zf), ls=ls, color=colour,
                        lw=1.8, zorder=2)
                handles[key] = Line2D([], [], color=colour, ls=ls, marker=marker,
                                      ms=5, label=rf'{short}: $x={-slope:.1f}$')
                continue

            # No 50% crossing: show the 16-84% spread of what it did select.
            zb, lo, hi = [], [], []
            for zv in np.unique(pop['z']):
                m = (pop['z'] == zv) & (pop['ffb'] == 1)
                if m.sum() < 10:
                    continue
                zb.append(zv)
                lo.append(np.percentile(pop['logmv'][m], 16))
                hi.append(np.percentile(pop['logmv'][m], 84))
            if len(zb) >= 3:
                ax.fill_between(zb, lo, hi, color=colour, alpha=0.20, lw=0, zorder=1)
                ax.plot(zb, lo, ls=ls, color=colour, lw=1.0, alpha=0.8, zorder=1)
                ax.plot(zb, hi, ls=ls, color=colour, lw=1.0, alpha=0.8, zorder=1)
                handles[key] = Line2D([], [], color=colour, ls=ls, lw=6, alpha=0.45,
                                      label=f'{short}: no threshold '
                                            r'(16--84\% band)')
            elif np.count_nonzero(pop['ffb'] == 1):
                # Too few to characterise at all -- say so rather than omit it.
                ax.plot(pop['z'][pop['ffb'] == 1], pop['logmv'][pop['ffb'] == 1],
                        marker, color=colour, ms=5, mfc='none', mew=1.2, zorder=3)
                handles[key] = Line2D([], [], color=colour, ls='none', marker=marker,
                                      ms=5, mfc='none', mew=1.2,
                                      label=f'{short}: only '
                                            f'{np.count_nonzero(pop["ffb"] == 1)} selected')

        # mode 8: median n_sh -> n_fbk, from the matching no-FFB run, because
        # its 50%-selection mass does not exist below z ~ 9.
        # Mode 8's threshold needs the matching no-FFB run.  Not every
        # simulation has one (there is no miniUchuu no-FFB), so skip that one
        # curve rather than failing the whole figure.
        noffb = dirs['_noffb']
        if not glob.glob(os.path.join(noffb, 'model_*.hdf5')):
            print(f'  threshold figure: no {noffb}, skipping the mode-8 curve '
                  f'for {label}')
            continue
        hdr = header(noffb)
        z, logmv, vvir, rvir, mdot = read_shell_inputs(noffb, zmin, zmax)
        n_sh = shell_density(mdot, rvir, vvir, clumping=1.0)
        short, colour, _mk = MODE_STYLE['mode8']
        zs, ms_thr, extrap = [], [], []
        for zv in np.unique(z):
            m = z == zv
            t, ex = threshold_from_shell_density(logmv[m], n_sh[m])
            if np.isfinite(t):
                zs.append(zv); ms_thr.append(t); extrap.append(ex)
        zs, ms_thr, extrap = np.array(zs), np.array(ms_thr), np.array(extrap)
        if zs.size >= 3:
            slope, inter = np.polyfit(np.log10(1 + zs), ms_thr, 1)
            ax.plot(zs[~extrap], ms_thr[~extrap], marker, color=colour, ms=5,
                    alpha=0.85, zorder=3)
            ax.plot(zs[extrap], ms_thr[extrap], marker, color=colour, ms=5,
                    mfc='white', mew=1.3, zorder=3)
            zf = np.linspace(zs.min(), zs.max(), 100)
            ax.plot(zf, inter + slope * np.log10(1 + zf), ls=ls, color=colour, lw=1.8,
                    zorder=2)
            handles['mode8'] = Line2D([], [], color=colour, ls=ls, marker=marker,
                                      ms=5, label=rf'{short}: $x={-slope:.1f}$')

    # the hardcoded Li+24 law itself, for reference
    zf = np.linspace(zmin, zmax, 200)
    ax.plot(zf, np.log10(ffb_threshold_mass_msun(zf)), color=THRESHOLD_RED, ls=':',
            lw=2.4, zorder=4)
    handles['_eq2'] = Line2D([], [], color=THRESHOLD_RED, ls=':', lw=2.4,
                             label=r'Li+24 eq. 2: $10^{10.8}[(1{+}z)/10]^{-6.2}$')

    ax.set_xlim(zmin, zmax)
    ax.set_xlabel(r'Redshift, $z$')
    ax.set_ylabel(r'Threshold halo mass, $\log_{10}(M_{\rm v,ffb}/M_\odot)$')
    ax.set_title('FFB threshold mass against redshift, all criteria\n'
                 r'fitted slope $x$ in $M_{\rm v,ffb} \propto (1+z)^{-x}$; '
                 'open symbols: extrapolated beyond the most massive halo present',
                 fontsize=10.5, color=INK, pad=10)
    ordered = [handles[k] for k, _t in MODES if k in handles]
    if '_eq2' in handles:
        ordered.append(handles['_eq2'])
    # Reserve just enough room under the x label for three legend rows; the
    # anchor sits at the top of that reserved strip so the gap stays small.
    fig.subplots_adjust(bottom=0.27)
    fig.legend(handles=ordered, loc='upper center', bbox_to_anchor=(0.5, 0.135),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.6,
               handletextpad=0.6, labelspacing=0.4)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBThresholds_MvirRedshift.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-root', default='output', metavar='DIR',
                    help='directory holding the run directories (default %(default)s). '
                         'On another machine this is typically where the runs were '
                         'staged, e.g. output/testmodels.')
    ap.add_argument('--sim-dirs', nargs='+', default=DEFAULT_SIM_DIRS, metavar='BASE',
                    help='base run-directory names, one per simulation (default: '
                         '%(default)s). Each criterion is looked up as BASE + a suffix '
                         'from MODE_DIR_SUFFIX, so e.g. "miniUchuu" resolves to '
                         'miniUchuu, miniUchuu_ffbmode8, miniUchuu_mbk_smooth and so on.')
    ap.add_argument('--naming', choices=sorted(NAMING_SCHEMES), default='local',
                    help="how run directories are named: 'local' (default) uses "
                         "BASE, BASE_mbk_smooth, BASE_noffb; 'hpc' uses BASE_FFB, "
                         'BASE_FFB_MBK25, BASE_noFFB. See NAMING_SCHEMES.')
    ap.add_argument('--mode-dir', action='append', default=[], metavar='KEY=SUFFIX',
                    help='override one entry of the naming scheme, e.g. '
                         'mode8=_FFB_SHELL. Repeatable. Keys: '
                         + ', '.join(sorted(NAMING_SCHEMES['local'])) + '.')
    ap.add_argument('--outdir', default=None,
                    help='where to write the figures (default: <output-root>/plots/)')
    ap.add_argument('--all-types', action='store_true',
                    help='include satellites as well as centrals. Off by default: mode 8 '
                         'is centrals-only in the model itself, so including satellites '
                         'would compare the three criteria on different galaxy sets -- '
                         "see read_selection()'s docstring.")
    ap.add_argument('--zmin', type=float, default=5.0)
    ap.add_argument('--zmax', type=float, default=14.0)
    ap.add_argument('--dilute', type=int, default=DILUTE_BACKGROUND, metavar='N',
                    help='cap the grey unselected background at N points per simulation '
                         'on the scatter figure (0 or unset to draw them all; default %(default)s)')
    ap.add_argument('--mv-bin-dex', type=float, default=MV_BIN_DEX, metavar='DEX',
                    help='log10(Mvir) bin height for the density maps (default %(default)g). '
                         'The redshift bins cannot be refined the same way: the output '
                         'snapshots are discrete, so one column per snapshot is already the '
                         'finest honest binning in z.')
    args = ap.parse_args()

    overrides = {}
    for item in args.mode_dir:
        if '=' not in item:
            raise SystemExit(f'--mode-dir expects KEY=SUFFIX, got {item!r}')
        key, suffix = item.split('=', 1)
        if key not in NAMING_SCHEMES[args.naming]:
            raise SystemExit(f'--mode-dir: unknown key {key!r}; expected one of '
                             + ', '.join(sorted(NAMING_SCHEMES[args.naming])))
        overrides[key] = suffix

    global SIMS
    SIMS = build_sims(args.sim_dirs, args.output_root, args.naming, overrides)
    if args.outdir is None:
        args.outdir = os.path.join(args.output_root, 'plots')

    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    print(f'FFBRegime == 1, read from the runs themselves '
          f"({'all galaxies' if args.all_types else 'centrals only'}):")
    by_sim = {label: {} for label, _c, _d in SIMS}
    by_mode = {key: [] for key, _t in MODES}
    for label, colour, dirs in SIMS:
        for key, _title in MODES:
            d = dirs[key]
            # Skip rather than crash on a run that has not been produced yet:
            # these figures get regenerated while new modes are still being
            # run, and losing the other eleven panels to one missing directory
            # is not a useful failure.
            if not glob.glob(os.path.join(d, 'model_*.hdf5')):
                print(f'  {label:12s} {key:6s}: SKIPPED, no model_*.hdf5 in {d}')
                continue
            pop = read_selection(d, header(d), args.zmin, args.zmax,
                                 centrals_only=not args.all_types)
            sel = pop['ffb'] == 1
            by_mode[key].append((label, colour, pop, sel))
            by_sim[label][key] = (pop, sel)
            print(f'  {label:12s} {key:6s}: {np.count_nonzero(sel):6d} / {sel.size:7d} '
                  f'selected ({100.0 * sel.mean():5.2f}%)')

    # Drop any mode no simulation could supply, so the panel count matches
    # what there is data for.
    modes_present = [(k, t) for k, t in MODES if by_mode[k]]
    if not modes_present:
        raise SystemExit('No runs found for any mode under output/.')

    zlim = (args.zmin, args.zmax)
    all_entries = [e for key, _t in modes_present for e in by_mode[key]]
    mvlim = (min(p['logmv'].min() for _l, _c, p, _s in all_entries) - 0.1,
             max(p['logmv'].max() for _l, _c, p, _s in all_entries) + 0.1)

    make_scatter_figure([(title, by_mode[key]) for key, title in modes_present],
                        args.outdir, zlim, mvlim, args.dilute)

    make_threshold_figure(args.outdir, args.zmin, args.zmax)

    make_fraction_figure(by_sim, args.outdir, zlim,
                         population_label=('all galaxies' if args.all_types
                                           else 'central galaxies'))

    # The density maps show only the SELECTED galaxies, so they get their own,
    # much tighter mass range -- shared across both simulations' figures so the
    # two can be read side by side, and padded to whole mv_bin_dex steps.
    # A criterion can legitimately select NOTHING (mode 10 on microUchuu does),
    # so skip empty selections here rather than taking min() of an empty array;
    # and fall back to the full population if nothing at all was selected.
    selected_masses = [p['logmv'][s] for modes in by_sim.values()
                       for p, s in modes.values() if np.any(s)]
    if selected_masses:
        sel_lo = min(a.min() for a in selected_masses)
        sel_hi = max(a.max() for a in selected_masses)
    else:
        sel_lo, sel_hi = mvlim
    density_mvlim = (np.floor((sel_lo - 0.1) / args.mv_bin_dex) * args.mv_bin_dex,
                     np.ceil((sel_hi + 0.1) / args.mv_bin_dex) * args.mv_bin_dex)

    make_density_figure(by_sim, modes_present, args.outdir, zlim, density_mvlim,
                        args.mv_bin_dex)


if __name__ == '__main__':
    main()

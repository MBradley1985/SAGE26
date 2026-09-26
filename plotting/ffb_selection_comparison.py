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

    FFBSelection_Density_millennium.pdf  the same comparison as a number-density
    FFBSelection_Density_microUchuu.pdf  map of the selected galaxies, one
                                         figure per simulation (two colour
                                         scales cannot share a panel).  The
                                         panels of a figure share one log
                                         colour scale, so the criteria are
                                         directly comparable within a
                                         simulation; the scale is NOT shared
                                         between the two figures, whose boxes
                                         and populations differ.

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
from matplotlib.colors import LogNorm

from ffb_cold_gas_density import (header, model_files, ffb_threshold_mass_msun,
                                   N_FBK_CM3, spin_parameter)

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'kieren_cohare_palatino_sty.mplstyle')

# One entry per simulation, in fixed order -- the colour follows the
# simulation, never its position in a panel, so a galaxy's colour means the
# same thing in both panels. Blue/orange is the canonical dichromat-safe pair;
# both were checked against each other and against the eq. 62 curve's red
# under deuteranopia/protanopia/tritanopia (OKLab dE: 32.9 normal, >=26.6 CVD
# between the two; 11.9 deuteranopic against the red curve) and for contrast
# on white (7.9:1 and 5.2:1).
SIMS = [
    ('millennium', '#1F45C0', dict(mode1='millennium',
                                   mbk='millennium_mbk_smooth',
                                   mode8='millennium_ffbmode8',
                                   mode9='millennium_ffbmode9',
                                   mode10='millennium_ffbmode10',
                                   mode11='millennium_ffbmode11')),
    ('microUchuu', '#A35A00', dict(mode1='microuchuu',
                                   mbk='microuchuu_mbk_smooth',
                                   mode8='microuchuu_ffbmode8',
                                   mode9='microuchuu_ffbmode9',
                                   mode10='microuchuu_ffbmode10',
                                   mode11='microuchuu_ffbmode11')),
]

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




def read_selection(directory, hdr, zmin, zmax, centrals_only=True,
                    min_particles=0, min_spin_percentile=0.0):
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

    min_particles and min_spin_percentile together drop the numerically
    unreliable tail (see --resolved).  get_disk_radius() in model_misc.c sets
    the disc radius straight from the halo's spin vector with no floor, so a
    poorly-resolved halo -- whose spin is shot-noise-dominated -- gets an
    arbitrarily small disc and an arbitrarily large density.  That matters far
    more for the Dekel+23 DISC criterion than for anything else here, since its
    3D condition scales as lambda^-13/6 and selects almost entirely on spin.
    The cut is analysis-side only; it changes no model output.
    """
    snaps = [s for s in hdr['output_snaps'] if zmin <= hdr['z'][s] <= zmax]
    z_l, logmv_l, ffb_l, spin_l = [], [], [], []
    for snap in snaps:
        mv_l, f_l, sp_l = [], [], []
        for path in model_files(directory):
            with h5py.File(path, 'r') as f:
                grp = f.get(f'Snap_{snap}')
                if grp is None:
                    continue
                mv = np.asarray(grp['Mvir'], dtype=np.float64) * hdr['to_msun']
                keep = mv > 0
                if centrals_only:
                    keep &= np.asarray(grp['Type'], dtype=np.int32) == 0
                if min_particles > 0:
                    keep &= np.asarray(grp['Len']) >= min_particles
                mv_l.append(mv[keep])
                f_l.append(np.asarray(grp['FFBRegime'], dtype=np.int32)[keep])
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
    if not z_l:
        raise SystemExit(f'No galaxies in {directory} for {zmin} <= z <= {zmax}.')

    pop = {'z': np.concatenate(z_l), 'logmv': np.concatenate(logmv_l),
           'ffb': np.concatenate(ffb_l), 'spin': np.concatenate(spin_l)}

    if min_spin_percentile > 0:
        finite = np.isfinite(pop['spin'])
        floor = np.percentile(pop['spin'][finite], min_spin_percentile)
        keep = finite & (pop['spin'] >= floor)
        pop = {k: v[keep] for k, v in pop.items()}

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

def make_density_figure(sim_label, pops, outdir, zlim, mvlim, mv_bin_dex,
                         style='grid', hex_gridsize=(13, 44)):
    """Number density of the SELECTED galaxies in (z, Mvir), one panel per
    criterion, on a shared log colour scale so the panels are comparable.

    pops is [(title, pop, selected), ...] in MODES order.

    style='grid' bins one column per output snapshot (see snapshot_z_edges())
    -- the finest honest binning in z, since every galaxy in a snapshot sits
    at exactly the same redshift.

    style='hex' lays hexagons over the plane instead. Offered, but NOT the
    default, and the reason is worth recording: hexbin wants both axes sampled
    comparably, and here z is 11-12 discrete snapshots against a continuous
    mass axis. Hexagons narrow enough to resolve mass fall inside individual
    snapshots and leave empty columns between them (the map breaks into
    disconnected dots); hexagons wide enough to bridge the spacing (nx ~ 13,
    since the median spacing is ~0.7-0.85 in z over 5<z<14) render as flat
    lozenges rather than hexagons. Either way it reads worse than one column
    per snapshot, and it additionally paints density at redshifts where
    nothing was sampled.
    """
    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('white')

    fig, axes = plt.subplots(1, len(pops), figsize=(6.4 * len(pops), 6.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.set_tight_layout(False)

    if style == 'hex':
        # Draw first with a placeholder norm, then restyle once every panel's
        # counts are known, so all panels share one scale.
        meshes = []
        for ax, (_title, pop, sel) in zip(axes, pops):
            meshes.append(ax.hexbin(pop['z'][sel], pop['logmv'][sel],
                                    gridsize=hex_gridsize, cmap=cmap, mincnt=1,
                                    extent=(*zlim, *mvlim), linewidths=0.0, zorder=2))
        vmax = max((m.get_array().max() for m in meshes if m.get_array().size), default=1)
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))
        for m in meshes:
            m.set_norm(norm)
        mesh = meshes[-1]
        bin_note = (f'hexagons, {hex_gridsize[0]}' r'$\times$' f'{hex_gridsize[1]} grid')
    else:
        z_edges = snapshot_z_edges(np.concatenate([p['z'] for _t, p, _s in pops]))
        mv_edges = np.arange(mvlim[0], mvlim[1] + mv_bin_dex, mv_bin_dex)
        grids = []
        for _title, pop, sel in pops:
            h, _, _ = np.histogram2d(pop['z'][sel], pop['logmv'][sel],
                                     bins=[z_edges, mv_edges])
            grids.append(np.ma.masked_where(h == 0, h))   # empty cells stay blank
        vmax = (max(g.max() for g in grids if g.count())
                if any(g.count() for g in grids) else 1)
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))
        for ax, grid in zip(axes, grids):
            mesh = ax.pcolormesh(z_edges, mv_edges, grid.T, cmap=cmap, norm=norm,
                                 shading='flat', zorder=2)
        bin_note = (r'bins of one snapshot $\times$ ' f'{mv_bin_dex:g} dex')

    for ax, (title, _pop, sel) in zip(axes, pops):
        # Cyan, not the scatter figure's red: plasma runs through red/orange at
        # its high end, where a red curve would vanish into the densest cells.
        threshold_curve(ax, zlim, THRESHOLD_CYAN)
        ax.set_xlim(*zlim)
        ax.set_ylim(*mvlim)
        ax.set_xlabel(r'Redshift, $z$')
        ax.set_title(f'{title}\n{np.count_nonzero(sel)} galaxies selected',
                     fontsize=10.5, color=INK, pad=8)

    axes[0].set_ylabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$')
    axes[-1].legend(loc='lower left', frameon=True, framealpha=0.85, fontsize=9)

    # subplots_adjust BEFORE the colorbar: fig.colorbar(ax=...) shrinks the
    # axes to make room for itself, and a later subplots_adjust would undo
    # that shrink and leave the bar sitting on top of the last panel.
    fig.subplots_adjust(top=0.80, wspace=0.06)
    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.035, pad=0.02)
    cbar.set_label('FFB-selected galaxies per bin')

    fig.suptitle(f'{sim_label}: FFB selection density, {bin_note}',
                 fontsize=12.5, color=INK, y=0.99)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'FFBSelection_Density_{sim_label}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}')


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


def make_threshold_figure(outdir, zmin, zmax, min_particles=0, min_spin_pct=0.0):
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
    from ffb_cold_gas_density import gather_population, shell_density

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    fig.set_tight_layout(False)
    handles = []

    for sim_i, (label, _colour, dirs) in enumerate(SIMS):
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
            rundir = f'output/{dirs[key]}'
            if not glob.glob(os.path.join(rundir, 'model_*.hdf5')):
                continue
            pop = read_selection(rundir, header(rundir), zmin, zmax,
                                 min_particles=min_particles,
                                 min_spin_percentile=min_spin_pct)
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
                handles.append(Line2D([], [], color=colour, ls=ls, marker=marker, ms=5,
                                      label=rf'{short} | {label} | $x={-slope:.1f}$'))
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
                handles.append(Line2D([], [], color=colour, ls=ls, lw=6, alpha=0.45,
                                      label=f'{short} | {label} | no threshold exists '
                                            r'(16--84\% of selections)'))
            elif np.count_nonzero(pop['ffb'] == 1):
                # Too few to characterise at all -- say so rather than omit it.
                ax.plot(pop['z'][pop['ffb'] == 1], pop['logmv'][pop['ffb'] == 1],
                        marker, color=colour, ms=5, mfc='none', mew=1.2, zorder=3)
                handles.append(Line2D([], [], color=colour, ls='none', marker=marker,
                                      ms=5, mfc='none', mew=1.2,
                                      label=f'{short} | {label} | only '
                                            f'{np.count_nonzero(pop["ffb"] == 1)} selected'))

        # mode 8: median n_sh -> n_fbk, from the matching no-FFB run, because
        # its 50%-selection mass does not exist below z ~ 9.
        noffb = f"output/{dirs['mode1'].split('_')[0]}_noffb"
        hdr = header(noffb)
        z, _ms, logmv, _n, _cg, _r, vvir, _f, rvir, mdot = gather_population(
            noffb, hdr, zmin, zmax, 0.0, 0.0, 0, with_accretion=True)
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
            handles.append(Line2D([], [], color=colour, ls=ls, marker=marker, ms=5,
                                  label=rf'{short} | {label} | $x={-slope:.1f}$'))

    # the hardcoded Li+24 law itself, for reference
    zf = np.linspace(zmin, zmax, 200)
    ax.plot(zf, np.log10(ffb_threshold_mass_msun(zf)), color=THRESHOLD_RED, ls=':',
            lw=2.4, zorder=4)
    handles.append(Line2D([], [], color=THRESHOLD_RED, ls=':', lw=2.4,
                          label=r'Li+24 eq. 2 as written: $10^{10.8}[(1{+}z)/10]^{-6.2}$'))

    ax.set_xlim(zmin, zmax)
    ax.set_xlabel(r'Redshift, $z$')
    ax.set_ylabel(r'Threshold halo mass, $\log_{10}(M_{\rm v,ffb}/M_\odot)$')
    ax.set_title('FFB threshold mass against redshift, all criteria\n'
                 r'fitted slope $x$ in $M_{\rm v,ffb} \propto (1+z)^{-x}$; '
                 'open symbols: extrapolated beyond the most massive halo present',
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=8.5,
              labelspacing=0.35)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'FFBThresholds_MvirRedshift.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=os.path.join('output', 'plots'),
                    help='default: output/plots/')
    ap.add_argument('--resolved', action='store_true',
                    help='drop the numerically unreliable tail before plotting: haloes '
                         'with fewer than --resolved-min-particles N-body particles, and '
                         'the bottom --resolved-spin-percentile per cent of the spin '
                         'distribution. One combined switch because the two select nearly '
                         'the same haloes -- a poorly-resolved halo has a shot-noise '
                         'spin vector, and get_disk_radius() turns that straight into an '
                         'arbitrarily small disc with no floor. Matters most for the '
                         'Dekel+23 disc criterion, whose 3D condition goes as '
                         'lambda^-13/6. ANALYSIS ONLY -- changes no model output.')
    ap.add_argument('--resolved-min-particles', type=int, default=20, metavar='N',
                    help='particle-count floor used by --resolved (default %(default)d)')
    ap.add_argument('--resolved-spin-percentile', type=float, default=1.0, metavar='PCT',
                    help='spin percentile floor used by --resolved (default %(default)g)')
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
    ap.add_argument('--density-style', choices=('grid', 'hex'), default='grid',
                    help="how the density maps bin the plane: 'grid' (default) uses one "
                         'column per output snapshot, the finest honest binning in z; '
                         "'hex' lays hexagons over it instead, which does not suit an "
                         'axis sampled at a dozen discrete redshifts -- see '
                         'make_density_figure() for what goes wrong.')
    ap.add_argument('--hex-gridsize', type=int, nargs=2, default=(13, 44),
                    metavar=('NX', 'NY'),
                    help='hexagon grid resolution for --density-style hex '
                         '(default %(default)s)')
    ap.add_argument('--mv-bin-dex', type=float, default=MV_BIN_DEX, metavar='DEX',
                    help='log10(Mvir) bin height for the density maps (default %(default)g). '
                         'The redshift bins cannot be refined the same way: the output '
                         'snapshots are discrete, so one column per snapshot is already the '
                         'finest honest binning in z.')
    args = ap.parse_args()

    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    min_particles = args.resolved_min_particles if args.resolved else 0
    min_spin_pct = args.resolved_spin_percentile if args.resolved else 0.0
    cuts = ('all galaxies' if args.all_types else 'centrals only')
    if args.resolved:
        cuts += (f'; Len >= {min_particles}, spin above the '
                 f'{min_spin_pct:g}th percentile')
    print(f'FFBRegime == 1, read from the runs themselves ({cuts}):')
    by_sim = {label: {} for label, _c, _d in SIMS}
    by_mode = {key: [] for key, _t in MODES}
    for label, colour, dirs in SIMS:
        for key, _title in MODES:
            d = os.path.join('output', dirs[key])
            # Skip rather than crash on a run that has not been produced yet:
            # these figures get regenerated while new modes are still being
            # run, and losing the other eleven panels to one missing directory
            # is not a useful failure.
            if not glob.glob(os.path.join(d, 'model_*.hdf5')):
                print(f'  {label:12s} {key:6s}: SKIPPED, no model_*.hdf5 in {d}')
                continue
            pop = read_selection(d, header(d), args.zmin, args.zmax,
                                 centrals_only=not args.all_types,
                                 min_particles=min_particles,
                                 min_spin_percentile=min_spin_pct)
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

    make_threshold_figure(args.outdir, args.zmin, args.zmax,
                          min_particles=min_particles, min_spin_pct=min_spin_pct)

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

    for label, _colour, _dirs in SIMS:
        make_density_figure(label,
                            [(title, *by_sim[label][key]) for key, title in modes_present
                             if key in by_sim[label]],
                            args.outdir, zlim, density_mvlim, args.mv_bin_dex,
                            style=args.density_style,
                            hex_gridsize=tuple(args.hex_gridsize))


if __name__ == '__main__':
    main()

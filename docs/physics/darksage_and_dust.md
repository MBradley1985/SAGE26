# Radially resolved discs and dust (DarkSAGEMode, DustySAGEMode)

Two optional layers of detail on top of the SAGE26 recipes. Both are off by
default (`DarkSAGEMode = 0`, `DustySAGEMode = 0`), and with both off the output
is identical to SAGE26. They work alone or together, and with every other
SAGE26 option. The parameters are in [Parameters](../parameters.md) and the
outputs in [Outputs](../outputs.md).

## DarkSAGEMode: discs in annuli of specific angular momentum

Follows DarkSage (Stevens, Croton & Mutch 2016; Stevens et al. 2024;
source `~/Documents/DarkSage/code`). Code: `src/model_darksage_disc.c`,
`model_darksage_sf.c`, `model_darksage_instability.c`,
`model_darksage_mergers.c`, and `darksage_deposit_cooling()` in
`model_cooling_heating.c`.

**Structure.** Each galaxy's gas and stellar disc is held in 30 annuli with
fixed specific angular momentum edges, j_0 = 0 and
j_i = 1 kpc/h km/s x 1.4^(i-1). Every snapshot, and every substep after the
first, `update_disc_radii()` maps the edges to radii through the galaxy's
rotation curve (NFW halo, beta-profile hot gas, Hernquist bulges and
intrahalo stars, the discs themselves, and the rotational support of the
disc). The annulus radii, potentials and circular velocities are refreshed
again at the end of each snapshot, so the written radii follow the final
masses, as in DarkSage.

**Angular momentum.** Cooling gas joins the annuli with an exponential
j-profile, j exp(-j/j_s); the gas-disc axis is the angular-momentum-weighted
mean of the hot gas and the existing disc, and a tilted disc is projected
onto the new axis. Counter-rotating gas collides and loses angular momentum.
The gas disc precesses towards the stellar disc. New stars join the stellar
disc by j. In minor mergers the satellite's gas joins the annuli its orbital
j reaches; in major mergers both discs are rebuilt from segments of each
annulus in the centre-of-momentum frame.

**Physics.** SAGE26 keeps deciding how much gas cools, forms stars, is
reheated, becomes unstable, feeds the black hole or bursts.
- Mode 1 shares those bulk masses over the annuli: star formation and H2 by
  the same law evaluated per annulus, the instability's gas and stars from
  the annuli the Toomre Q marks unstable. The annuli sum to the bulk fields.
  `DiskScaleRadius` stays at SAGE26's halo-spin value, so every shared output
  is identical to SAGE26: SAGE26 with radially resolved discs and angular
  momentum tracking, and nothing else changed.
- `DarkSAGERadialFlowsOn` (any mode) adds DarkSage's radial flows inside the
  disc: after SAGE26's instability has taken its mass, annuli still Toomre
  unstable pass 66% of their unstable gas (1) and also their unstable stars
  (2, DarkSage's 1 - 0.34 c_s / sigma) to the neighbouring annuli, from the
  outside in, with the disc's angular momentum kept. Only the annuli change.
  Without it the annuli have no radial flow.
- Mode 2 is mode 1 with `DiskScaleRadius` measured from the annuli: half the
  mass-weighted mean radius of the gas and disc stars, using the disc's own
  rotation speed. This is the one change it makes to SAGE26's physics.
- Mode 3 is mode 2 with the star-formation law evaluated on each annulus's
  own surface density; the galaxy's star formation is the sum.

**Known behaviour** (mode 1, Millennium, star-forming disc centrals of
10^10-10^11 Msun). Stellar metallicity rises outwards (about 0.65 to 1.0 Zsun
from 0.25 to 2 stellar half-mass radii): the discs grow inside out, the inner
stars formed when the gas was poorer, and the gas metallicity rises about
threefold after z ~ 2, more than its radial gradient. `DarkSAGERadialFlowsOn 2`
flattens it (0.9 to 0.95 Zsun) but does not make it fall outwards, and shrinks
the annulus stellar half-mass radius by about 25%; gas alone (1) moves little
gas, as SAGE26's instability has already taken the unstable gas. The gas
metallicity follows SAGE26's (`Yield` 0.025).

**Known behaviour** (modes 2 and 3). The stellar discs are more compact than observed
(about 0.3 dex below Lange et al. 2016 at z = 0), as in DarkSage's own runs
on the same trees. Black holes in galaxies below 10^9.5 Msun are about
0.4 dex heavier than in SAGE26, matching DarkSage. Gas-rich small discs come
out larger than SAGE26's, so they form stars less efficiently: the cosmic star
formation rate density is 0.1-0.15 dex lower at z > 2 and converges with
SAGE26 below z ~ 1, and small haloes hold up to ~0.1 dex less stellar mass.
All of these enter through the disc scale radius (modes 2 and 3) and are
absent in mode 1.

## DustySAGEMode: dust

Follows Dusty-SAGE (Triani et al. 2020; source `PhD/dusty-sage/src`). Code:
`src/model_dust.c`, and the dust transport helpers in `src/model_misc.h`.

**Reservoirs.** Cold, hot, CGM and ejected dust. Dust moves with the gas at
the dust-to-gas ratio of the reservoir the gas leaves, wherever SAGE26 moves
gas (cooling, star formation, feedback, reincorporation, infall, stripping,
mergers, disruption).

**Processes,** once per galaxy per substep after star formation and mergers
(`dust_evolve()`):
- Production by AGB stars and SNII from delayed element yields over the
  galaxy's star-formation history (Arrigoni et al. 2010 IMF; Raiteri et al.
  1996 lifetimes; Karakas 2010, Woosley & Weaver 1995 or Nomoto et al. 2006,
  Iwamoto et al. 1999 or Seitenzahl et al. 2013 yields), with Popping et al.
  (2017) condensation. SNIa dust is off, as in Dusty-SAGE.
- Grain growth in molecular gas (Asano et al. 2013 eq. 20), stopping at the
  dust-to-metals ratio `DustGrowthMaxDtoZ` (0.5, where the source stops). It
  needs H2, so it is off with `SFprescription` 0 or 2.
- Destruction by SN shocks (Asano et al. 2013).
- Thermal sputtering in the hot, CGM and ejected gas of centrals and
  satellites (Popping et al. 2017).

**Known behaviour.** Above about Zsun, grain growth outpaces destruction and
holds the dust-to-metals ratio near the ceiling: ~0.42 in mode 1 disc
centrals of 10^10-10^11 Msun (dust-to-gas ~0.01). With `DustGrowthMaxDtoZ 1`
(Asano as written) it reaches 0.8-0.9, the dust-to-gas doubles and the z = 0
cold-dust density is ~0.3 dex higher.

**Modes.**
- Mode 1, passive: SAGE26's metals are unchanged and the dust is part of
  them; production is capped by the metals SAGE26 made. Every non-dust output
  is identical to mode 0.
- Mode 2: the delayed yields replace SAGE26's instantaneous yield, and the
  dust is held outside `MetalsColdGas` (metals + dust are conserved). Stars
  forming from dusty gas take its dust into their metals; the cooling
  metallicity includes the halo's dust.

**Changes from the Dusty-SAGE source.** Fixed: the yield metallicity bin
(zero-metallicity yields were always used), the yield lookup time (truncated
to an integer), the IMF (missing square), the Nomoto metallicity grid (an
octal constant), and three errors that switched sputtering off. Changed: the
yields are made once per substep (the source made them again in every merger
burst); snapshot times and the growth, destruction and sputtering timescales
carry their factors of h; the yields read the current snapshot's mean star
formation rate so far; growth is limited by all the metals, with the
source's stopping point (dust-to-metals 0.5) kept as `DustGrowthMaxDtoZ`;
satellites' hot dust is sputtered; and a merged satellite's star-formation
history joins the central's. Grains are floored at 0.1% of their initial
size.

## Both together

With both modes on, the cold dust is carried by the annuli (`DiscDust`,
summing to `ColdDust`). It moves with the annulus gas in every annulus
operation, new stars and reheated gas take the dust of their own annulus,
the yields go to the annuli forming stars (else the stellar disc, else the
gas), and grain growth and SN destruction act on each annulus's gas, metals,
dust, H2 and star formation. DarkSAGEMode 1-2 share the galaxy-wide dust rates
over the annuli; DarkSAGEMode 3 sums the annuli's own. Neither source has
dust in annuli, so this combination follows the DarkSAGEMode pattern above.

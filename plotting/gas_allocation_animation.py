#!/usr/bin/env python3
"""
gas_allocation_animation.py
============================
Animate how a single galaxy's cold gas is allocated each snapshot -- split
between star formation, SN reheating (cold -> hot), and black hole
accretion -- following the joint cold-gas budget introduced in
model_mergers.c / model_starformation_and_feedback.c.

Per-step allocation is reconstructed from fields SAGE already saves:
  * StarsFormed  = (SfrDisk + SfrBulge) * dT          [Msun, this step]
  * SNreheated   = OutflowRate * dT                    [Msun, this step]
  * BHaccreted   = QuasarModeBHaccretionMass            [Msun, this step;
                    mass the black hole accreted during the last timestep]
  * ColdGas      = remaining cold gas reservoir         [Msun]
(dT, the time since this galaxy was last evolved, converts the two rate
fields from Msun/yr to a step mass; see find_galaxy.py's docstring for the
shared unit conventions this reuses.)

The galaxy is tracked across snapshots by GalaxyIndex (or whichever ID field
find_galaxy.find_id_field locates), exactly as find_galaxy.py's
--all-snapshots mode does.

Output is an animated GIF (via Pillow) by default, one frame per snapshot the
galaxy exists in; --format mp4 uses ffmpeg if available.

Examples
--------
  python gas_allocation_animation.py -i "output/millennium_mini/model_*.hdf5"
  python gas_allocation_animation.py -i "..." --galaxy-id 12345 -o bh.gif
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_galaxy import (read_simulation_params, read_snapshot_frame,
                          find_id_field, get_redshift)

SEC_PER_YEAR_IN_MYR = 1.0e6

COMPONENT_COLORS = {
    'Stars formed': '#4C72B0',
    'SN reheated': '#DD8452',
    'BH accreted': '#55A868',
}


def select_default_galaxy(file_list, sim, id_field):
    """Pick the galaxy with the largest black hole mass at the last snapshot."""
    snap_num = sim['latest_snapshot']
    frames = []
    for fpath in file_list:
        df = read_snapshot_frame(fpath, snap_num, sim['Hubble_h'],
                                  sim['UnitTime_in_s'], os.path.basename(fpath))
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    latest = pd.concat(frames, ignore_index=True)
    if latest.empty or 'BlackHoleMass' not in latest.columns:
        return None
    row = latest.loc[latest['BlackHoleMass'].idxmax()]
    return int(row[id_field])


def load_history(file_list, sim, id_field, galaxy_id):
    frames = []
    for snap_num in sim['available_snapshots']:
        for fpath in file_list:
            df = read_snapshot_frame(fpath, snap_num, sim['Hubble_h'],
                                      sim['UnitTime_in_s'], os.path.basename(fpath))
            if df is None or id_field not in df.columns:
                continue
            match = df[df[id_field] == galaxy_id]
            if not match.empty:
                frames.append(match)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values('Snapshot').reset_index(drop=True)
    history['Redshift'] = [get_redshift(int(s), sim['redshifts'])
                            for s in history['Snapshot']]
    return history


def compute_allocation(history):
    dt_yr = history['dT'].clip(lower=0) * SEC_PER_YEAR_IN_MYR
    history['StarsFormed'] = (history['SfrDisk'] + history['SfrBulge']).clip(lower=0) * dt_yr
    history['SNreheated'] = history['OutflowRate'].clip(lower=0) * dt_yr
    history['BHaccreted'] = history['QuasarModeBHaccretionMass'].clip(lower=0)
    return history


def make_animation(history, galaxy_id, out_path, fps, fmt):
    components = ['StarsFormed', 'SNreheated', 'BHaccreted']
    labels = ['Stars formed', 'SN reheated', 'BH accreted']
    colors = [COMPONENT_COLORS[l] for l in labels]

    z = history['Redshift'].to_numpy()
    cold_gas = history['ColdGas'].to_numpy()
    n_frames = len(history)

    fig, (ax_hist, ax_pie) = plt.subplots(1, 2, figsize=(11, 5),
                                           gridspec_kw={'width_ratios': [1.6, 1]})

    # Left panel: cumulative log-scale trends up to the current snapshot.
    max_y = max(1.0, np.nanmax([history[c].max() for c in components] + [cold_gas.max()]))
    ax_hist.set_xlim(z.max() + 0.2 * (z.max() - z.min() + 1e-6), z.min() - 0.2 * (z.max() - z.min() + 1e-6))
    ax_hist.set_ylim(1e-2, max_y * 3)
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel('Redshift')
    ax_hist.set_ylabel(r'Mass this step [M$_\odot$]')
    ax_hist.set_title(f'Galaxy ID {galaxy_id}: allocation history')

    lines = {}
    for comp, label, color in zip(components, labels, colors):
        (line,) = ax_hist.plot([], [], color=color, label=label, lw=2)
        lines[comp] = line
    (cold_line,) = ax_hist.plot([], [], color='0.4', ls='--', lw=1.5, label='ColdGas reservoir')
    marker = ax_hist.axvline(z[0], color='k', lw=1, alpha=0.6)
    ax_hist.legend(loc='upper left', fontsize=8)

    ax_pie.set_title('This step\'s allocation')

    def init():
        for comp in components:
            lines[comp].set_data([], [])
        cold_line.set_data([], [])
        return list(lines.values()) + [cold_line, marker]

    def update(frame_idx):
        for comp in components:
            lines[comp].set_data(z[:frame_idx + 1], history[comp].to_numpy()[:frame_idx + 1])
        cold_line.set_data(z[:frame_idx + 1], cold_gas[:frame_idx + 1])
        marker.set_xdata([z[frame_idx], z[frame_idx]])

        ax_pie.clear()
        ax_pie.set_title('This step\'s allocation')
        shares = [max(history[comp].to_numpy()[frame_idx], 0.0) for comp in components]
        total = sum(shares)
        snap = int(history['Snapshot'].to_numpy()[frame_idx])
        zval = z[frame_idx]
        subtitle = f"Snap {snap}"
        if zval is not None and not np.isnan(zval):
            subtitle += f"  (z = {zval:.2f})"
        if total <= 0:
            ax_pie.text(0.5, 0.5, "No allocation\nthis step", ha='center', va='center',
                        transform=ax_pie.transAxes)
        else:
            ax_pie.pie(shares, labels=labels, colors=colors, autopct='%1.0f%%',
                      startangle=90)
        ax_pie.text(0.5, -0.15, subtitle, ha='center', va='center',
                    transform=ax_pie.transAxes, fontsize=9)
        return list(lines.values()) + [cold_line, marker]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                                    blit=False, interval=1000 / fps)

    if fmt == 'mp4':
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Animate a single SAGE26 galaxy's cold-gas allocation "
                    "(stars / SN reheating / BH accretion) across snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium_mini/model_*.hdf5',
                   help='Glob for the model HDF5 files.')
    p.add_argument('--galaxy-id', type=int, default=None,
                   help='Galaxy ID to track (default: pick the largest '
                        'black hole at the latest snapshot).')
    p.add_argument('--id-field', default=None,
                   help='Override auto-detected ID field.')
    p.add_argument('-o', '--output', default=None,
                   help='Output animation path (default: '
                        'gas_allocation_<galaxy_id>.<gif|mp4> next to the '
                        'first input file).')
    p.add_argument('--format', choices=['gif', 'mp4'], default='gif',
                   help='Animation format (default: gif; mp4 needs ffmpeg).')
    p.add_argument('--fps', type=float, default=4.0,
                   help='Frames per second (default: 4).')
    args = p.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: no files match {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    if not sim['available_snapshots']:
        print("Error: no Snap_<N> groups found in the input files.")
        sys.exit(1)

    id_field = find_id_field(file_list, sim['latest_snapshot'], args.id_field)
    if id_field is None:
        print("Error: no galaxy-ID field found in the input files.")
        sys.exit(1)

    galaxy_id = args.galaxy_id
    if galaxy_id is None:
        galaxy_id = select_default_galaxy(file_list, sim, id_field)
        if galaxy_id is None:
            print("Error: could not auto-select a galaxy (no data at the "
                  "latest snapshot).")
            sys.exit(1)
        print(f"No --galaxy-id given; auto-selected {id_field} = {galaxy_id} "
              "(largest BlackHoleMass at the latest snapshot).")

    history = load_history(file_list, sim, id_field, galaxy_id)
    if history.empty:
        print(f"Error: no records found for {id_field} = {galaxy_id}.")
        sys.exit(1)
    if len(history) < 2:
        print(f"Error: {id_field} = {galaxy_id} only appears in "
              f"{len(history)} snapshot(s); need at least 2 to animate.")
        sys.exit(1)

    history = compute_allocation(history)

    input_dir = os.path.dirname(os.path.abspath(file_list[0]))
    out_path = args.output or os.path.join(
        input_dir, f"gas_allocation_{galaxy_id}.{args.format}")

    print(f"Tracking {id_field} = {galaxy_id} across {len(history)} snapshots.")
    print(f"Rendering {args.format.upper()} -> {out_path}")
    make_animation(history, galaxy_id, out_path, args.fps, args.format)
    print("Done.")


if __name__ == "__main__":
    main()

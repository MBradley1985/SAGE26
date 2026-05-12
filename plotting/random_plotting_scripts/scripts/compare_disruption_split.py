#!/usr/bin/env python
"""
Compare two SAGE runs: fixed vs dynamic disruption split.
Focuses on quantities affected by how disrupted satellite mass
is partitioned between ICL and BCG.
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import glob

# ========================== CONFIGURATION ==========================

RUNS = {
    'Millennium Dynamic Split + concentration alpha': {
        'dir': './output/millennium/',
        'color': 'steelblue',
    },
    'Millennium Fixed Split': {
        'dir': './output/millennium_fixedsplit/',
        'color': 'forestgreen',
    },
    'Millennium Dynamic Split': {
        'dir': './output/millennium_dynsplit/',
        'color': 'darkorange',
    },
}

OUTPUT_DIR = './plots/'

MASS_BINS = {
    'Groups':   (10**12.5, 10**14),
    'Clusters': (10**14,   np.inf),
}

# ========================== I/O ==========================

def read_header(directory):
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model files in {directory}")
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        runtime = f['Header/Runtime']
        header = {
            'hubble_h':     float(sim.attrs['hubble_h']),
            'mass_convert': float(runtime.attrs['UnitMass_in_g']) / 1.989e33
                            / float(sim.attrs['hubble_h']),
            'redshifts':    list(f['Header/snapshot_redshifts'][:]),
            'output_snaps': sorted(int(s) for s in f['Header/output_snapshots'][:]),
            'DynamicDisruptionSplit': int(runtime.attrs.get('DynamicDisruptionSplit', 0)),
            'DisruptionSplitAlpha':   float(runtime.attrs.get('DisruptionSplitAlpha', 0.4)),
            'FractionDisruptedToICS': float(runtime.attrs.get('FractionDisruptedToICS', 0.8)),
            'DisruptionSplitCref':   float(runtime.attrs.get('DisruptionSplitCref', 10.0)),
        }
    return header, files


def read_z0(directory):
    """Read z=0 galaxy properties from all model files."""
    header, files = read_header(directory)
    mc = header['mass_convert']
    snap = header['output_snaps'][-1]
    snap_key = f'Snap_{snap}'

    props = ['Type', 'StellarMass', 'BulgeMass', 'MergerBulgeMass',
             'InstabilityBulgeMass', 'IntraClusterStars',
             'ICS_disrupt', 'ICS_accrete',
             'CentralMvir', 'BlackHoleMass', 'Mvir',
             'GalaxyIndex', 'CentralGalaxyIndex']
    mass_props = {'StellarMass', 'BulgeMass', 'MergerBulgeMass',
                  'InstabilityBulgeMass', 'IntraClusterStars',
                  'ICS_disrupt', 'ICS_accrete',
                  'CentralMvir', 'BlackHoleMass', 'Mvir'}

    chunks = {p: [] for p in props}
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for p in props:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))

    data = {}
    for p in props:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            if p in mass_props:
                arr *= mc
            data[p] = arr

    return data, header


# ========================== PLOTTING ==========================

def setup_style():
    style = {
        'figure.dpi': 150,
        'font.size': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
    }
    plt.rcParams.update(style)


def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Line styles: solid for clusters, dashed for groups
    mass_bin_ls = {'Clusters': '-', 'Groups': '--'}

    # Load data for each run
    run_data = {}
    for label, cfg in RUNS.items():
        data, header = read_z0(cfg['dir'])
        ds = header['DynamicDisruptionSplit']
        alpha = header['DisruptionSplitAlpha']
        frac = header['FractionDisruptedToICS']
        cref = header['DisruptionSplitCref']
        print(f"{label}: DynamicSplit={ds}, alpha={alpha}, f_ICS={frac}, c_ref={cref}")

        is_central = data['Type'] == 0
        selections = {}
        for mb_label, (lo, hi) in MASS_BINS.items():
            sel = is_central & (data['CentralMvir'] >= lo) & (data['CentralMvir'] < hi)
            selections[mb_label] = sel
            print(f"  {mb_label}: {np.sum(sel):,} centrals")

        run_data[label] = {
            'data': data,
            'header': header,
            'selections': selections,
            'cfg': cfg,
        }

    # ----------------------------------------------------------------
    # Figure: 2x2 panel comparison
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel (a): Central stellar mass distribution ---
    ax = axes[0, 0]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        for mb_label, sel in rd['selections'].items():
            bcg_mass = d['StellarMass'][sel]
            bcg_mass = bcg_mass[bcg_mass > 0]
            if len(bcg_mass) == 0:
                continue
            log_m = np.log10(bcg_mass)
            ax.hist(log_m, bins=20, histtype='step', color=cfg['color'],
                    ls=mass_bin_ls[mb_label], lw=2, density=True)
    ax.set_xlabel(r'$\log_{10}(M_{*,\rm central}\;/\;M_\odot)$')
    ax.set_ylabel('Normalised count')
    ax.set_title('(a) Central stellar mass')

    # --- Panel (b): ICL mass distribution ---
    ax = axes[0, 1]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        for mb_label, sel in rd['selections'].items():
            ics = d['IntraClusterStars'][sel]
            ics = ics[ics > 0]
            if len(ics) == 0:
                continue
            log_ics = np.log10(ics)
            ax.hist(log_ics, bins=20, histtype='step', color=cfg['color'],
                    ls=mass_bin_ls[mb_label], lw=2, density=True)
    ax.set_xlabel(r'$\log_{10}(M_{\rm ICL}\;/\;M_\odot)$')
    ax.set_ylabel('Normalised count')
    ax.set_title('(b) ICL mass')

    # --- Panel (c): ICL fraction vs halo mass ---
    ax = axes[1, 0]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        # Combine all centrals across mass bins for this panel
        all_sel = np.zeros(len(d['Type']), dtype=bool)
        for sel in rd['selections'].values():
            all_sel |= sel

        # f_ICL = ICS / total stellar mass in group/cluster
        # Sum satellite stellar masses per central
        central_gidx = d['GalaxyIndex'][all_sel]
        mvir = d['CentralMvir'][all_sel]
        ics = d['IntraClusterStars'][all_sel]
        central_smass = d['StellarMass'][all_sel]

        # Build lookup: central GalaxyIndex -> sum of all member stellar masses
        central_set = set(central_gidx)
        sat_smass_sum = {cg: 0.0 for cg in central_gidx}
        all_cgidx = d['CentralGalaxyIndex']
        all_smass = d['StellarMass']
        all_type = d['Type']
        for j in range(len(all_cgidx)):
            if all_type[j] != 0 and all_cgidx[j] in central_set:
                sat_smass_sum[all_cgidx[j]] += all_smass[j]
        sat_sum = np.array([sat_smass_sum[cg] for cg in central_gidx])

        total = central_smass + sat_sum + ics
        valid = total > 0
        f_icl = np.where(valid, ics / total, 0)

        log_mvir = np.log10(mvir[valid])
        f_icl = f_icl[valid]

        # Binned median
        bins = np.linspace(12.5, max(log_mvir.max(), 14.5), 10)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        idx = np.digitize(log_mvir, bins)
        medians = np.array([np.median(f_icl[idx == i]) if np.any(idx == i) else np.nan
                            for i in range(1, len(bins))])
        p16 = np.array([np.percentile(f_icl[idx == i], 16) if np.sum(idx == i) > 2 else np.nan
                         for i in range(1, len(bins))])
        p84 = np.array([np.percentile(f_icl[idx == i], 84) if np.sum(idx == i) > 2 else np.nan
                         for i in range(1, len(bins))])

        ok = np.isfinite(medians)
        ax.plot(bin_centres[ok], medians[ok], color=cfg['color'],
                lw=2, marker='o', label=label)
        ax.fill_between(bin_centres[ok], p16[ok], p84[ok],
                         color=cfg['color'], alpha=0.15)

    ax.axvline(14.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax.text(14.02, 0.02, 'groups | clusters', fontsize=8, color='grey',
            rotation=90, va='bottom', transform=ax.get_xaxis_transform())
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}\;/\;M_\odot)$')
    ax.set_ylabel(r'$f_{\rm ICL} = M_{\rm ICL} \,/\, M_{*,\rm total}$')
    ax.set_title('(c) ICL fraction vs halo mass')
    ax.legend(fontsize=10)

    # --- Panel (d): Bulge mass breakdown (merger vs instability vs disruption) ---
    ax = axes[1, 1]
    categories = ['Merger', 'Instability', 'Disruption']
    x_pos = np.arange(len(categories))
    n_runs = len(RUNS)
    bar_width = 0.8 / (n_runs * 2)  # 2 mass bins per run
    bar_i = 0
    for mb_label in MASS_BINS:
        for label, rd in run_data.items():
            d = rd['data']
            cfg = rd['cfg']
            sel = rd['selections'][mb_label]
            total_bulge = np.sum(d['BulgeMass'][sel])
            if total_bulge <= 0:
                bar_i += 1
                continue
            merger_bulge = np.sum(d['MergerBulgeMass'][sel])
            instab_bulge = np.sum(d['InstabilityBulgeMass'][sel])
            disrupt_bulge = total_bulge - merger_bulge - instab_bulge

            vals = np.array([merger_bulge, instab_bulge, disrupt_bulge])
            vals /= total_bulge
            offset = -0.4 + bar_width * (bar_i + 0.5)
            hatch = '//' if mb_label == 'Groups' else None
            ax.bar(x_pos + offset, vals, bar_width, color=cfg['color'],
                   alpha=0.7, edgecolor='black', linewidth=0.5, hatch=hatch)
            bar_i += 1

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Fraction of total bulge mass')
    ax.set_title('(d) Central bulge mass origin')

    # Shared legend for all panels
    legend_elements = [
        Line2D([0], [0], color=cfg['color'], lw=2.5, label=label)
        for label, cfg in RUNS.items()
    ] + [
        Line2D([0], [0], color='grey', lw=2, ls='-', label='Clusters'),
        Line2D([0], [0], color='grey', lw=2, ls='--', label='Groups'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(RUNS) + 2, fontsize=10,
               framealpha=0.9, columnspacing=1.5, handletextpad=0.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    outpath = os.path.join(OUTPUT_DIR, 'disruption_split_comparison.pdf')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Figure 2: B/T vs stellar mass
    # ----------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel (a): Total B/T vs M* ---
    ax = axes2[0]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        for mb_label, sel in rd['selections'].items():
            sm = d['StellarMass'][sel]
            valid = sm > 0
            log_sm = np.log10(sm[valid])
            bt = d['BulgeMass'][sel][valid] / sm[valid]

            bins = np.linspace(log_sm.min(), log_sm.max(), 10)
            bc = 0.5 * (bins[:-1] + bins[1:])
            idx = np.digitize(log_sm, bins)
            med = np.array([np.median(bt[idx == i]) if np.any(idx == i) else np.nan
                            for i in range(1, len(bins))])
            p16 = np.array([np.percentile(bt[idx == i], 16) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            p84 = np.array([np.percentile(bt[idx == i], 84) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            ok = np.isfinite(med)
            ax.plot(bc[ok], med[ok], color=cfg['color'], ls=mass_bin_ls[mb_label],
                    lw=2, marker='o', markersize=5)
            ax.fill_between(bc[ok], p16[ok], p84[ok], color=cfg['color'], alpha=0.1)
    ax.set_xlabel(r'$\log_{10}(M_{*}\;/\;M_\odot)$')
    ax.set_ylabel(r'$B/T$')
    ax.set_title('(a) Total B/T')
    ax.set_ylim(0, 1)

    # --- Panel (b): Merger B/T vs M* ---
    ax = axes2[1]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        for mb_label, sel in rd['selections'].items():
            sm = d['StellarMass'][sel]
            valid = sm > 0
            log_sm = np.log10(sm[valid])
            mbt = d['MergerBulgeMass'][sel][valid] / sm[valid]

            bins = np.linspace(log_sm.min(), log_sm.max(), 10)
            bc = 0.5 * (bins[:-1] + bins[1:])
            idx = np.digitize(log_sm, bins)
            med = np.array([np.median(mbt[idx == i]) if np.any(idx == i) else np.nan
                            for i in range(1, len(bins))])
            p16 = np.array([np.percentile(mbt[idx == i], 16) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            p84 = np.array([np.percentile(mbt[idx == i], 84) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            ok = np.isfinite(med)
            ax.plot(bc[ok], med[ok], color=cfg['color'], ls=mass_bin_ls[mb_label],
                    lw=2, marker='o', markersize=5)
            ax.fill_between(bc[ok], p16[ok], p84[ok], color=cfg['color'], alpha=0.1)
    ax.set_xlabel(r'$\log_{10}(M_{*}\;/\;M_\odot)$')
    ax.set_ylabel(r'$M_{\rm bulge,merger} \,/\, M_*$')
    ax.set_title('(b) Merger bulge fraction')
    ax.set_ylim(0, 1)

    # --- Panel (c): Instability B/T vs M* ---
    ax = axes2[2]
    for label, rd in run_data.items():
        d = rd['data']
        cfg = rd['cfg']
        for mb_label, sel in rd['selections'].items():
            sm = d['StellarMass'][sel]
            valid = sm > 0
            log_sm = np.log10(sm[valid])
            ibt = d['InstabilityBulgeMass'][sel][valid] / sm[valid]

            bins = np.linspace(log_sm.min(), log_sm.max(), 10)
            bc = 0.5 * (bins[:-1] + bins[1:])
            idx = np.digitize(log_sm, bins)
            med = np.array([np.median(ibt[idx == i]) if np.any(idx == i) else np.nan
                            for i in range(1, len(bins))])
            p16 = np.array([np.percentile(ibt[idx == i], 16) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            p84 = np.array([np.percentile(ibt[idx == i], 84) if np.sum(idx == i) > 2 else np.nan
                            for i in range(1, len(bins))])
            ok = np.isfinite(med)
            ax.plot(bc[ok], med[ok], color=cfg['color'], ls=mass_bin_ls[mb_label],
                    lw=2, marker='o', markersize=5)
            ax.fill_between(bc[ok], p16[ok], p84[ok], color=cfg['color'], alpha=0.1)
    ax.set_xlabel(r'$\log_{10}(M_{*}\;/\;M_\odot)$')
    ax.set_ylabel(r'$M_{\rm bulge,instab} \,/\, M_*$')
    ax.set_title('(c) Instability bulge fraction')
    ax.set_ylim(0, 1)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=cfg['color'], lw=2.5, label=label)
        for label, cfg in RUNS.items()
    ] + [
        Line2D([0], [0], color='grey', lw=2, ls='-', label='Clusters'),
        Line2D([0], [0], color='grey', lw=2, ls='--', label='Groups'),
    ]
    fig2.legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=(0.5, 0.0), ncol=len(RUNS) + 2, fontsize=10,
                framealpha=0.9, columnspacing=1.5, handletextpad=0.5)

    fig2.tight_layout(rect=[0, 0.07, 1, 1])
    outpath2 = os.path.join(OUTPUT_DIR, 'bulge_fraction_comparison.pdf')
    fig2.savefig(outpath2, dpi=200, bbox_inches='tight')
    print(f"Saved: {outpath2}")
    plt.close(fig2)

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY (z=0 centrals)")
    print("=" * 70)
    for mb_label, (lo, hi) in MASS_BINS.items():
        hi_str = f'{np.log10(hi):.1f}' if np.isfinite(hi) else 'inf'
        print(f"\n  {'='*30} {mb_label} ({np.log10(lo):.1f} < log Mvir < {hi_str}) {'='*30}")
        for label, rd in run_data.items():
            d = rd['data']
            sel = rd['selections'][mb_label]
            n = np.sum(sel)
            if n == 0:
                print(f"\n  --- {label} ---")
                print(f"  N centrals: 0")
                continue
            central_smass = d['StellarMass'][sel]
            ics = d['IntraClusterStars'][sel]
            bulge = d['BulgeMass'][sel]
            merger_b = d['MergerBulgeMass'][sel]
            instab_b = d['InstabilityBulgeMass'][sel]
            disrupt_b = bulge - merger_b - instab_b
            ics_d = d['ICS_disrupt'][sel]
            ics_a = d['ICS_accrete'][sel]

            # Total stellar mass = central + satellites + ICL
            central_gidx_s = d['GalaxyIndex'][sel]
            cset = set(central_gidx_s)
            sat_sum_s = {cg: 0.0 for cg in central_gidx_s}
            for j in range(len(d['CentralGalaxyIndex'])):
                if d['Type'][j] != 0 and d['CentralGalaxyIndex'][j] in cset:
                    sat_sum_s[d['CentralGalaxyIndex'][j]] += d['StellarMass'][j]
            sat_arr = np.array([sat_sum_s[cg] for cg in central_gidx_s])
            total = central_smass + sat_arr + ics

            has_stars = central_smass > 0
            safe_sm = np.where(has_stars, central_smass, 1.0)
            bt = np.where(has_stars, bulge / safe_sm, np.nan)
            mbt = np.where(has_stars, merger_b / safe_sm, np.nan)
            ibt = np.where(has_stars, instab_b / safe_sm, np.nan)
            dbt = np.where(has_stars, disrupt_b / safe_sm, np.nan)

            n_zero = np.sum(~has_stars)
            print(f"\n  --- {label} ---")
            print(f"  N centrals:           {n:,}" +
                  (f"  ({n_zero} with M*=0)" if n_zero > 0 else ""))
            print(f"  Central M* median:    {np.median(central_smass):.3e} Msun")
            print(f"  ICL median:           {np.median(ics):.3e} Msun")
            print(f"  f_ICL median:         {np.median(ics / total):.3f}")
            print(f"  B/T median:           {np.nanmedian(bt):.3f}")
            print(f"    merger B/T:         {np.nanmedian(mbt):.3f}")
            print(f"    instability B/T:    {np.nanmedian(ibt):.3f}")
            print(f"    disruption B/T:     {np.nanmedian(dbt):.3f}")
            print(f"  ICS_disrupt total:    {np.sum(ics_d):.3e} Msun")
            print(f"  ICS_accrete total:    {np.sum(ics_a):.3e} Msun")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

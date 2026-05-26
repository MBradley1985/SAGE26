#!/usr/bin/env python3
"""
Radial integration vs fixed-area H2 prescription comparison.

Compares three Millennium runs:
  millennium      — H2RadialIntegrationOn=1 (radial integration)
  millennium_pir2 — H2DiskAreaOption=0 (pi*rs^2)
  millennium_pi3r2 — H2DiskAreaOption=1 (pi*(3rs)^2)

Run from the SAGE26 root:
    python plotting/random_plotting_scripts/radial_integration_comparison.py
"""

import os, glob
import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SNAP_KEY = 'Snap_49'   # z=0 for Millennium

MODELS = [
    {'dir': './output/microuchuu/',       'label': 'Radial integration', 'color': 'steelblue',   'ls': '-'},
    {'dir': './output/microuchuu_pir2/',  'label': r'Fixed $\pi r_s^2$', 'color': 'tomato',      'ls': '--'},
    {'dir': './output/microuchuu_pi3r2/', 'label': r'Fixed $\pi(3r_s)^2$','color': 'forestgreen','ls': '-.'},
        {'dir': './output/microuchuu_h2/',    'label': 'FFBs H2',            'color': 'mediumpurple','ls': ':'},
]

MSTAR_MIN = 1e8  # Msun

# ---------------------------------------------------------------------------
# Observational references (log space throughout)
# ---------------------------------------------------------------------------

# xCOLD GASS (Saintonge+2017): log(M_H2/M*) at z~0
COLDGASS_LOGMSTAR        = np.array([9.0,  9.5,  10.0, 10.5, 11.0])
COLDGASS_LOG_H2_FRAC_MED = np.array([-0.80,-1.00,-1.30,-1.70,-2.10])
COLDGASS_LOG_H2_FRAC_LO  = np.array([-1.00,-1.30,-1.70,-2.10,-2.50])
COLDGASS_LOG_H2_FRAC_HI  = np.array([-0.50,-0.70,-1.00,-1.30,-1.70])

# xGASS (Catinella+2018): log(M_HI/M*) at z~0
XGASS_LOGMSTAR          = np.array([9.0,  9.5,  10.0, 10.5, 11.0])
XGASS_LOG_HI_FRAC_MED   = np.array([ 0.10,-0.15,-0.50,-0.90,-1.40])
XGASS_LOG_HI_FRAC_LO    = np.array([-0.20,-0.50,-0.90,-1.30,-1.80])
XGASS_LOG_HI_FRAC_HI    = np.array([ 0.40, 0.20,-0.10,-0.50,-1.00])

def sfms_speagle(logmstar):
    t = 13.8
    return (0.84 - 0.026*t)*logmstar - (6.51 - 0.11*t)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_snap(directory, snap_key):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        print(f"  WARNING: no model_*.hdf5 in {directory}")
        return None, None, None

    with h5py.File(files[0], 'r') as f:
        h   = float(f['Header/Simulation'].attrs['hubble_h'])
        umg = float(f['Header/Runtime'].attrs['UnitMass_in_g'])

    mc = umg / 1.989e33 / h   # code mass → physical Msun

    fields = ['Type', 'StellarMass', 'BulgeMass', 'ColdGas', 'H2gas', 'H1gas',
              'SfrDisk', 'SfrBulge', 'DiskRadius', 'Mvir', 'MetalsColdGas']
    arrays = {k: [] for k in fields}

    for fp in files:
        with h5py.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for k in fields:
                if k in grp:
                    arrays[k].append(grp[k][:])

    if not any(arrays[k] for k in fields):
        print(f"  WARNING: {snap_key} not found in {directory}")
        return None, None, None

    data = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    return data, mc, h


def prepare(data, mc, h):
    mstar  = data['StellarMass'] * mc
    mbulge = data['BulgeMass']   * mc
    mh2    = data['H2gas']       * mc
    mhi    = data['H1gas']       * mc
    mcold  = data['ColdGas']     * mc
    sfr    = data['SfrDisk'] + data['SfrBulge']   # Msun/yr
    rs_kpc = data['DiskRadius'] * 1000.0 / h / 3.0  # Mpc/h → physical kpc, /3 = scale radius

    sel = (data['Type'] == 0) & (mstar > MSTAR_MIN) & np.isfinite(mstar)
    return {
        'mstar':  mstar[sel],  'mbulge': mbulge[sel], 'mh2':   mh2[sel],
        'mhi':    mhi[sel],    'mcold':  mcold[sel],   'sfr':   sfr[sel],
        'rs_kpc': rs_kpc[sel],
    }


def bin_stat(x, y, bins, min_n=20):
    mids, meds, p16, p84 = [], [], [], []
    for i in range(len(bins)-1):
        ok = (x >= bins[i]) & (x < bins[i+1]) & np.isfinite(y) & (y > 0)
        if ok.sum() >= min_n:
            mids.append(0.5*(bins[i]+bins[i+1]))
            meds.append(np.median(y[ok]))
            p16.append(np.percentile(y[ok], 16))
            p84.append(np.percentile(y[ok], 84))
    return map(np.array, (mids, meds, p16, p84))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(model_bm, model_log_med, obs_x, obs_log_med, obs_log_lo, obs_log_hi):
    """
    Compare model binned medians (log space) to observational reference.

    Returns chi2, fraction of bins within the observational 16-84% band,
    and signed mean offset in dex.

    sigma_obs is estimated as (P84 - P16) / 2, floored at 0.05 dex to
    avoid division by implausibly tight observational uncertainties.
    """
    if len(model_bm) < 2:
        return dict(chi2=np.nan, frac_in=np.nan, offset=np.nan, n_bins=0)

    in_range = (obs_x >= model_bm.min()) & (obs_x <= model_bm.max())
    if in_range.sum() < 2:
        return dict(chi2=np.nan, frac_in=np.nan, offset=np.nan, n_bins=0)

    ox    = obs_x[in_range]
    omed  = obs_log_med[in_range]
    olo   = obs_log_lo[in_range]
    ohi   = obs_log_hi[in_range]

    mod   = np.interp(ox, model_bm, model_log_med)
    sigma = np.clip((ohi - olo) / 2.0, 0.05, None)

    chi2    = float(np.sum((mod - omed)**2 / sigma**2))
    frac_in = float(np.mean((mod >= olo) & (mod <= ohi)))
    offset  = float(np.mean(mod - omed))

    return dict(chi2=chi2, frac_in=frac_in, offset=offset, n_bins=int(in_range.sum()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading models at {SNAP_KEY} ...")
    loaded = []
    for m in MODELS:
        data, mc, h = load_snap(m['dir'], SNAP_KEY)
        if data is None:
            continue
        gal = prepare(data, mc, h)
        loaded.append({**m, 'gal': gal})
        print(f"  {m['label']:30s}  {gal['mstar'].size:,} central galaxies")

    if not loaded:
        raise RuntimeError("No models loaded.")

    mass_bins = np.arange(8.5, 12.0, 0.5)
    size_bins = np.linspace(0, 30, 13)

    # -----------------------------------------------------------------------
    # Summary statistics — global medians
    # -----------------------------------------------------------------------
    print(f"\n{'Model':30s}  {'med H2/M*':>10}  {'med f_H2':>9}  {'med tdep(Gyr)':>13}")
    for m in loaded:
        g = m['gal']
        lm = np.log10(g['mstar'])
        sel = (lm > 9.5) & (lm < 11.0) & (g['mh2'] > 0) & (g['mcold'] > 0)
        f_h2 = g['mh2'] / np.clip(g['mcold'] * 0.74, 1e-30, None)
        tdep = np.where(g['sfr'] > 0, g['mh2'] / g['sfr'] / 1e9, np.nan)
        print(f"  {m['label']:30s}  "
              f"{np.median(g['mh2'][sel]/g['mstar'][sel]):10.4f}  "
              f"{np.median(f_h2[sel]):9.4f}  "
              f"{np.nanmedian(tdep[sel]):13.2f}")

    # -----------------------------------------------------------------------
    # Figure 1 — comparison plots
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.32)
    ax_h2, ax_hi, ax_fh2 = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_sfms, ax_tdep, ax_size = [fig.add_subplot(gs[1, c]) for c in range(3)]

    # Observational references
    ax_h2.fill_between(COLDGASS_LOGMSTAR, COLDGASS_LOG_H2_FRAC_LO, COLDGASS_LOG_H2_FRAC_HI,
                       color='grey', alpha=0.25, zorder=0)
    ax_h2.plot(COLDGASS_LOGMSTAR, COLDGASS_LOG_H2_FRAC_MED, 'k--', lw=1.5,
               label='xCOLD GASS (Saintonge+17)', zorder=1)

    ax_hi.fill_between(XGASS_LOGMSTAR, XGASS_LOG_HI_FRAC_LO, XGASS_LOG_HI_FRAC_HI,
                       color='grey', alpha=0.25, zorder=0)
    ax_hi.plot(XGASS_LOGMSTAR, XGASS_LOG_HI_FRAC_MED, 'k--', lw=1.5,
               label='xGASS (Catinella+18)', zorder=1)

    sfms_x = np.linspace(8.5, 11.5, 100)
    ax_sfms.plot(sfms_x, sfms_speagle(sfms_x), 'k--', lw=1.5, label='Speagle+14', zorder=1)

    ax_tdep.axhline(2.0, color='k', ls='--', lw=1.5, label='Obs ~2 Gyr', zorder=1)
    ax_tdep.axhspan(1.0, 3.0, color='grey', alpha=0.15, zorder=0)

    # Model curves — collect binned medians for stats
    stats_records = []

    for m in loaded:
        g   = m['gal']
        lm  = np.log10(g['mstar'])
        col = m['color']
        ls  = m['ls']
        lbl = m['label']
        f_h2 = np.clip(g['mh2'] / np.clip(g['mcold'] * 0.74, 1e-30, None), 0, 1)

        # H2/M* — collect in log space for stats
        bm, med, lo, hi = bin_stat(lm, g['mh2']/g['mstar'], mass_bins)
        h2_bm, h2_log_med = np.array([]), np.array([])
        if len(bm):
            h2_bm      = bm
            h2_log_med = np.log10(np.clip(med, 1e-6, None))
            ax_h2.plot(bm, h2_log_med, ls, color=col, lw=2, label=lbl)
            ax_h2.fill_between(bm, np.log10(np.clip(lo,1e-6,None)),
                               np.log10(np.clip(hi,1e-6,None)), color=col, alpha=0.12)

        # HI/M* — collect in log space for stats
        bm, med, lo, hi = bin_stat(lm, g['mhi']/g['mstar'], mass_bins)
        hi_bm, hi_log_med = np.array([]), np.array([])
        if len(bm):
            hi_bm      = bm
            hi_log_med = np.log10(np.clip(med, 1e-6, None))
            ax_hi.plot(bm, hi_log_med, ls, color=col, lw=2, label=lbl)
            ax_hi.fill_between(bm, np.log10(np.clip(lo,1e-6,None)),
                               np.log10(np.clip(hi,1e-6,None)), color=col, alpha=0.12)

        # f_H2 vs M*
        bm, med, lo, hi = bin_stat(lm, f_h2, mass_bins)
        if len(bm):
            ax_fh2.plot(bm, med, ls, color=col, lw=2, label=lbl)
            ax_fh2.fill_between(bm, lo, hi, color=col, alpha=0.12)

        # SFR main sequence
        sf = g['sfr'] > 0
        bm, med, lo, hi = bin_stat(lm[sf], np.log10(g['sfr'][sf]), mass_bins)
        if len(bm):
            ax_sfms.plot(bm, med, ls, color=col, lw=2, label=lbl)
            ax_sfms.fill_between(bm, lo, hi, color=col, alpha=0.12)

        # Depletion time
        ok = (g['sfr'] > 0) & (g['mh2'] > 0)
        tdep = g['mh2'][ok] / g['sfr'][ok] / 1e9
        bm, med, lo, hi = bin_stat(lm[ok], tdep, mass_bins)
        if len(bm):
            ax_tdep.plot(bm, med, ls, color=col, lw=2, label=lbl)
            ax_tdep.fill_between(bm, lo, hi, color=col, alpha=0.12)

        # f_H2 vs disk scale radius
        bm, med, lo, hi = bin_stat(g['rs_kpc'], f_h2, size_bins)
        if len(bm):
            ax_size.plot(bm, med, ls, color=col, lw=2, label=lbl)
            ax_size.fill_between(bm, lo, hi, color=col, alpha=0.12)

        # Compute χ² stats
        s_h2 = compute_stats(h2_bm, h2_log_med,
                             COLDGASS_LOGMSTAR, COLDGASS_LOG_H2_FRAC_MED,
                             COLDGASS_LOG_H2_FRAC_LO, COLDGASS_LOG_H2_FRAC_HI)
        s_hi = compute_stats(hi_bm, hi_log_med,
                             XGASS_LOGMSTAR, XGASS_LOG_HI_FRAC_MED,
                             XGASS_LOG_HI_FRAC_LO, XGASS_LOG_HI_FRAC_HI)
        stats_records.append({
            'label': lbl, 'color': col,
            'h2': s_h2, 'hi': s_hi,
            'chi2_total': s_h2['chi2'] + s_hi['chi2'],
        })

    # Axes formatting
    ax_h2.set_xlabel(r'$\log(M_*/M_\odot)$');  ax_h2.set_ylabel(r'$\log(M_{\rm H2}/M_*)$')
    ax_h2.set_xlim(8.5, 11.5);  ax_h2.set_ylim(-3.5, 0.5)
    ax_h2.legend(fontsize=8);   ax_h2.grid(True, alpha=0.3)
    ax_h2.set_title('H2 fraction vs stellar mass')

    ax_hi.set_xlabel(r'$\log(M_*/M_\odot)$');  ax_hi.set_ylabel(r'$\log(M_{\rm HI}/M_*)$')
    ax_hi.set_xlim(8.5, 11.5);  ax_hi.set_ylim(-3.5, 1.0)
    ax_hi.legend(fontsize=8);   ax_hi.grid(True, alpha=0.3)
    ax_hi.set_title('HI fraction vs stellar mass')

    ax_fh2.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax_fh2.set_ylabel(r'$f_{\rm H2} = M_{\rm H2}/(0.74\,M_{\rm cold})$')
    ax_fh2.set_xlim(8.5, 11.5);  ax_fh2.set_ylim(0, 1)
    ax_fh2.legend(fontsize=8);   ax_fh2.grid(True, alpha=0.3)
    ax_fh2.set_title('Molecular fraction vs stellar mass')

    ax_sfms.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax_sfms.set_ylabel(r'$\log({\rm SFR})\,[M_\odot\,{\rm yr}^{-1}]$')
    ax_sfms.set_xlim(8.5, 11.5);  ax_sfms.set_ylim(-3, 2)
    ax_sfms.legend(fontsize=8);   ax_sfms.grid(True, alpha=0.3)
    ax_sfms.set_title('Star-forming main sequence')

    ax_tdep.set_xlabel(r'$\log(M_*/M_\odot)$')
    ax_tdep.set_ylabel(r'$\tau_{\rm dep} = M_{\rm H2}/{\rm SFR}$ [Gyr]')
    ax_tdep.set_xlim(8.5, 11.5);  ax_tdep.set_ylim(0, 12)
    ax_tdep.legend(fontsize=8);   ax_tdep.grid(True, alpha=0.3)
    ax_tdep.set_title('H2 depletion time vs stellar mass')

    ax_size.set_xlabel(r'Disk scale radius $r_s$ [kpc]')
    ax_size.set_ylabel(r'$f_{\rm H2} = M_{\rm H2}/(0.74\,M_{\rm cold})$')
    ax_size.set_xlim(0, 25);  ax_size.set_ylim(0, 1)
    ax_size.legend(fontsize=8);  ax_size.grid(True, alpha=0.3)
    ax_size.set_title('Molecular fraction vs disk size')

    fig.suptitle(f'Radial integration vs fixed area  ({SNAP_KEY}, centrals, $M_* > 10^8 M_\\odot$)',
                 fontsize=12, y=1.01)
    plt.savefig('./output/radial_integration_comparison.pdf', bbox_inches='tight', dpi=150)
    print(f"\nSaved → ./output/radial_integration_comparison.pdf")
    plt.close()

    # -----------------------------------------------------------------------
    # Print χ² table (ranked by total χ²)
    # -----------------------------------------------------------------------
    stats_records.sort(key=lambda r: r['chi2_total'])

    print(f"\n{'— Observational comparison (χ² on binned medians) —':^80}")
    print(f"  σ_obs = (P84−P16)/2 per bin, floored at 0.05 dex")
    print(f"  n_bins: H2={stats_records[0]['h2']['n_bins']}, HI={stats_records[0]['hi']['n_bins']}\n")
    hdr = f"  {'Model':30s}  {'χ²_H2':>8}  {'χ²_HI':>8}  {'χ²_total':>10}  {'H2 in-band':>11}  {'HI in-band':>11}  {'H2 offset':>10}  {'HI offset':>10}"
    print(hdr)
    print('  ' + '-'*(len(hdr)-2))
    for r in stats_records:
        h2 = r['h2'];  hi = r['hi']
        print(f"  {r['label']:30s}  "
              f"{h2['chi2']:8.2f}  {hi['chi2']:8.2f}  {r['chi2_total']:10.2f}  "
              f"{h2['frac_in']*100:10.0f}%  {hi['frac_in']*100:10.0f}%  "
              f"{h2['offset']:+10.3f}  {hi['offset']:+10.3f}")
    print(f"\n  (offset > 0 = model above observations in log space)")

    # -----------------------------------------------------------------------
    # Figure 2 — statistical summary
    # -----------------------------------------------------------------------
    labels_clean = [r['label'] for r in stats_records]
    colors       = [r['color'] for r in stats_records]
    n = len(stats_records)
    y = np.arange(n)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, max(3, n*1.2 + 1.5)))
    fig2.suptitle('Observational alignment score  (lower χ², higher in-band = better)',
                  fontsize=11)

    # χ² breakdown
    ax_chi = axes2[0]
    chi2_h2  = [r['h2']['chi2'] for r in stats_records]
    chi2_hi  = [r['hi']['chi2'] for r in stats_records]
    chi2_tot = [r['chi2_total']  for r in stats_records]
    bar_h = 0.25

    ax_chi.barh(y + bar_h, chi2_h2,  bar_h, color=colors, alpha=0.9, label=r'$\chi^2_{\rm H2}$',  hatch='')
    ax_chi.barh(y,          chi2_hi,  bar_h, color=colors, alpha=0.5, label=r'$\chi^2_{\rm HI}$',  hatch='//')
    ax_chi.barh(y - bar_h,  chi2_tot, bar_h, color=colors, alpha=0.3, label=r'$\chi^2_{\rm total}$', hatch='xx')
    ax_chi.set_yticks(y);  ax_chi.set_yticklabels(labels_clean, fontsize=9)
    ax_chi.set_xlabel(r'$\chi^2$');  ax_chi.set_title(r'$\chi^2$ (lower = better)')
    ax_chi.legend(fontsize=8);  ax_chi.grid(True, alpha=0.3, axis='x')
    ax_chi.invert_yaxis()

    # Fraction of bins within observational 16–84% band
    ax_frac = axes2[1]
    frac_h2 = [r['h2']['frac_in']*100 for r in stats_records]
    frac_hi = [r['hi']['frac_in']*100 for r in stats_records]
    ax_frac.barh(y + bar_h/2, frac_h2, bar_h, color=colors, alpha=0.9, label='H2 vs xCOLD GASS')
    ax_frac.barh(y - bar_h/2, frac_hi, bar_h, color=colors, alpha=0.5, label='HI vs xGASS', hatch='//')
    ax_frac.axvline(100, color='grey', ls=':', lw=1)
    ax_frac.set_yticks(y);  ax_frac.set_yticklabels(labels_clean, fontsize=9)
    ax_frac.set_xlabel('% bins within obs. 16–84% band')
    ax_frac.set_title('In-band fraction (higher = better)')
    ax_frac.set_xlim(0, 110)
    ax_frac.legend(fontsize=8);  ax_frac.grid(True, alpha=0.3, axis='x')
    ax_frac.invert_yaxis()

    # Signed offset (dex)
    ax_off = axes2[2]
    off_h2 = [r['h2']['offset'] for r in stats_records]
    off_hi = [r['hi']['offset'] for r in stats_records]
    ax_off.barh(y + bar_h/2, off_h2, bar_h, color=colors, alpha=0.9, label='H2 offset')
    ax_off.barh(y - bar_h/2, off_hi, bar_h, color=colors, alpha=0.5, label='HI offset', hatch='//')
    ax_off.axvline(0, color='k', lw=1)
    ax_off.set_yticks(y);  ax_off.set_yticklabels(labels_clean, fontsize=9)
    ax_off.set_xlabel('Mean offset [dex]  (+ = model above obs)')
    ax_off.set_title('Systematic offset')
    ax_off.legend(fontsize=8);  ax_off.grid(True, alpha=0.3, axis='x')
    ax_off.invert_yaxis()

    plt.tight_layout()
    plt.savefig('./output/radial_integration_stats.pdf', bbox_inches='tight', dpi=150)
    print(f"Saved → ./output/radial_integration_stats.pdf")
    plt.close()


if __name__ == '__main__':
    main()

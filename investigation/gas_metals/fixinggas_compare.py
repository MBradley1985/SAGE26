"""Compare two SAGE26 runs (e.g. FixingGas = 0 and 1) against observations.

Usage: python fixinggas_compare.py <plot_dir> <prefix> <ref_dir> <ref_label> <dir> <label> [<dir> <label> ...]
The first run is the reference (FixingGas 0); the others are FixingGas levels.
Writes six multi-panel figures (law check, z = 0 gas, z = 0 population,
metallicities, redshift evolution, bursts and bulges) into <plot_dir>.
"""
import sys, os, h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from style import G05, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, OBS, DATA
from common import binned
from fig_fixes import speagle_ms_ssfr

PLOTS, PREFIX = sys.argv[1:3]
PAIRS = list(zip(sys.argv[3::2], sys.argv[4::2]))
os.makedirs(PLOTS, exist_ok=True)
HE = 1.36
EDGES = np.arange(8.5, 12.01, 0.2)
COLOURS = [VIOLET, ORANGE, AQUA, MAGENTA, YELLOW]
FIELDS = ['Type', 'StellarMass', 'BulgeMass', 'ColdGas', 'MetalsColdGas', 'MetalsStellarMass', 'H2gas', 'H1gas',
          'SfrDisk', 'SfrBulge', 'BlackHoleMass', 'H2DepletionTime_Gyr', 'HotGas', 'Mvir', 'MergerBulgeMass', 'InstabilityBulgeMass']


class Run:
    def __init__(self, d):
        self.f = h5py.File(os.path.join(d, 'model_0.hdf5'), 'r')
        self.h = float(self.f['Header/Simulation'].attrs['hubble_h'])
        self.box = float(self.f['Header/Simulation'].attrs['box_size'])
        self.z = np.array(self.f['Header/snapshot_redshifts'])
        self.vol = (self.box / self.h) ** 3
        self.cache = {}

    def snap_at(self, z):
        return int(np.argmin(np.abs(self.z - z)))

    def get(self, snap):
        if snap not in self.cache:
            g = self.f[f'Snap_{snap}']
            d = {k: g[k][:].astype('f8') for k in FIELDS}
            for k in ('StellarMass', 'BulgeMass', 'ColdGas', 'MetalsColdGas', 'MetalsStellarMass', 'H2gas', 'H1gas',
                      'BlackHoleMass', 'HotGas', 'Mvir', 'MergerBulgeMass', 'InstabilityBulgeMass'):
                d[k] *= 1e10 / self.h
            d['sfr'] = d['SfrDisk'] + d['SfrBulge']
            d['lm'] = np.log10(np.maximum(d['StellarMass'], 1.0))
            d['ssfr'] = d['sfr'] / np.maximum(d['StellarMass'], 1.0)
            d['z'] = self.z[snap]
            d['dms'] = d['ssfr'] / speagle_ms_ssfr(d['lm'], d['z'])
            d['ms'] = np.abs(np.log10(np.maximum(d['dms'], 1e-30))) < 0.4      # Tacconi/Saintonge-style MS cut
            d['sf'] = d['ssfr'] > 1e-11
            self.cache[snap] = d
        return self.cache[snap]


RUNS = [(Run(d), COLOURS[i], lab) for i, (d, lab) in enumerate(PAIRS)]
on = RUNS[-1][0]
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 2})


def lg(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)


def med(ax, x, y, sel, col, lab, ls='-', stat=np.median, shade=True, edges=EDGES):
    c, o = binned(x, y, edges, sel, stat=stat)
    ax.plot(c, o[:, 0], color=col, ls=ls, label=lab)
    if shade:
        ax.fill_between(c, o[:, 1], o[:, 2], color=col, alpha=0.12, lw=0)
    return c, o


def tacconi_tdep_gyr(lm, dms, z):
    return 10 ** (0.09 - 0.62 * np.log10(1 + z) - 0.44 * np.log10(np.clip(dms, 1e-2, 1e2)) + 0.09 * (lm - 10.7))


s0 = on.snap_at(0.0)

# ------------------------------------------------------------------ 1. law check
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
t1, t2 = [], []
for run, col, lab in RUNS[1:]:
    g = run.get(s0); sel = g['sf'] & (g['lm'] > 9) & (g['H2DepletionTime_Gyr'] > 0) & (g['H2gas'] > 0)
    if sel.sum() < 10:      # run without the Tacconi depletion time
        continue
    ratio = g['H2DepletionTime_Gyr'] / tacconi_tdep_gyr(g['lm'], g['dms'], g['z'])
    axs[0, 0].hist(np.log10(ratio[sel]), bins=np.linspace(-0.5, 0.5, 81), color=col, alpha=0.5, label=lab)
    r2 = g['SfrDisk'] / (HE * g['H2gas'] / (g['H2DepletionTime_Gyr'] * 1e9)); s2_ = sel & (g['SfrDisk'] > 0)
    axs[0, 1].hist(np.log10(r2[s2_]), bins=np.linspace(-1, 1, 81), color=col, alpha=0.5, label=lab)
    t1.append(f'{np.median(ratio[sel]):.3f}'); t2.append(f'{np.median(r2[s2_]):.3f}')
ax = axs[0, 0]; ax.axvline(0, color=OBS, lw=1); ax.legend(loc='upper left')
ax.set_xlabel(r'log [ $t_{\rm dep}$ used in SAGE26 / Tacconi+18 at output sSFR ]'); ax.set_ylabel('galaxies (SF, z = 0)')
ax.set_title(f'(a) t_dep matches Tacconi+18: median ratio {", ".join(t1)}', fontsize=10)
ax = axs[0, 1]; ax.axvline(0, color=OBS, lw=1); ax.legend(loc='upper left')
ax.set_xlabel(r'log [ SfrDisk / (1.36 $M_{\rm H_2}$ / $t_{\rm dep}$) ]'); ax.set_ylabel('galaxies (SF, z = 0)')
ax.set_title(f'(b) SFR follows the law: median {", ".join(t2)}\n'
             '(spread: SFR is the snapshot average, H$_2$ the end state)', fontsize=9)
ax = axs[1, 0]
lm_ = np.linspace(9, 11.5, 50)
for run, col, lab in RUNS:
    d = run.get(s0); ms = d['ms'] & (d['lm'] > 9) & (d['H2gas'] > 0) & (d['sfr'] > 0)
    med(ax, d['lm'], lg(HE * d['H2gas'] / d['sfr']), ms, col, lab)
ax.plot(lm_, np.log10(tacconi_tdep_gyr(lm_, 1.0, 0.0) * 1e9), color=OBS, ls=':', lw=1.5, label='Tacconi+18, z = 0 MS')
ax.axhspan(8.98 - 0.27, 8.98 + 0.27, color=OBS, alpha=0.08, lw=0, label='Saintonge+17 MS, 8.98 ± 0.27')
ax.set_ylabel(r'log $t_{\rm dep}({\rm H_2+He})$ [yr], MS galaxies'); ax.set_ylim(8.3, 10.2); ax.legend(loc='upper left')
ax.set_title('(c) molecular depletion time, z = 0', fontsize=10); ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_xlim(9, 11.6)
ax = axs[1, 1]
zs = [0, 0.5, 1, 1.5, 2, 3, 4]
for run, col, lab in RUNS:
    vals = []
    for z in zs:
        d = run.get(run.snap_at(z)); m = d['ms'] & (np.abs(d['lm'] - 10.25) < 0.25) & (d['H2gas'] > 0) & (d['sfr'] > 0)
        vals.append(np.median(lg(HE * d['H2gas'][m] / d['sfr'][m])) if m.sum() > 10 else np.nan)
    ax.plot(np.log10(1 + np.array(zs)), vals, 'o-', color=col, label=lab)
zz = np.linspace(0, 4, 40)
ax.plot(np.log10(1 + zz), np.log10(tacconi_tdep_gyr(10.25, 1.0, zz) * 1e9), color=OBS, ls=':', lw=1.5, label='Tacconi+18, MS, log M* = 10.25')
ax.set_xlabel('log (1 + z)'); ax.set_ylabel(r'log $t_{\rm dep}({\rm H_2+He})$ [yr]'); ax.legend(loc='lower left')
ax.set_title('(d) depletion time with redshift, MS, log M* = 10–10.5', fontsize=10)
fig.suptitle('Check 1: the Tacconi+18 star-formation law is applied as written (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_1_law_check.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ 2. gas at z = 0
fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.4))
logmean = lambda v: np.log10(np.mean(10 ** v[np.isfinite(v)]))
ax = axs[0, 0]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(HE * d['H2gas'] / d['StellarMass']), d['ms'] & (d['lm'] > 9), col, lab, stat=logmean, shade=False)
s17 = np.array([[10.07, -0.89], [10.25, -1.05], [10.46, -0.99], [10.68, -1.10], [10.82, -1.26], [11.10, -1.41]])
ax.plot(s17[:, 0], s17[:, 1], 's', color=OBS, label='xCOLD GASS MS stacks (Saintonge+17 T5)')
ax.plot(lm_, -1.19 - 0.36 * (lm_ - 10.7), color=OBS, ls=':', lw=1.5, label=r'Tacconi+18 $\mu_{\rm gas}$, z = 0 MS')
ax.set_ylabel(r'log $\langle M_{\rm H_2+He}/M_*\rangle$, MS'); ax.set_ylim(-2.2, 0.3); ax.legend(loc='lower left')
ax.set_title('(a) molecular gas, main sequence', fontsize=10)
ax = axs[0, 1]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['H1gas'] / d['StellarMass']), d['lm'] > 9, col, lab)
x = np.loadtxt(DATA + 'Gas/HIGasRatio_NonDetEQUpperLimits.dat')
ax.errorbar(x[:, 0], x[:, 1], yerr=[x[:, 2], x[:, 3]], fmt='o', color=OBS, ms=5, capsize=2, lw=1, label='xGASS medians (non-det = limit)')
c18 = np.array([[10.07, -0.869], [10.34, -1.175], [10.65, -1.231], [10.95, -1.475], [11.20, -1.589]])
ax.plot(c18[:, 0], c18[:, 1], 'D', color=OBS, mfc='white', label='xGASS means (Catinella+18 T1)')
ax.set_ylabel(r'log $M_{\rm HI}/M_*$, all galaxies'); ax.set_ylim(-3, 1); ax.legend(loc='lower left')
ax.set_title('(b) atomic gas, all galaxies', fontsize=10)
ax = axs[1, 0]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(HE * d['H2gas'] / d['StellarMass']), d['lm'] > 9, col, lab)
x = np.loadtxt(DATA + 'Gas/MolecularGasRatio_NonDetEQUpperLimits.dat')
ax.errorbar(x[:, 0], x[:, 1], yerr=[x[:, 2], x[:, 3]], fmt='o', color=OBS, ms=5, capsize=2, lw=1, label='xCOLD GASS medians (non-det = limit)')
ax.set_ylabel(r'log $M_{\rm H_2+He}/M_*$, all galaxies'); ax.set_ylim(-3.5, 0.5); ax.legend(loc='lower left')
ax.set_title('(c) molecular gas, all galaxies', fontsize=10)
ax = axs[1, 1]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['ColdGas'] / d['StellarMass']), d['lm'] > 9, col, lab)
ax.plot(lm_, -0.48 * lm_ + 4.39, color=OBS, ls='--', lw=1.5, label='Peeples+14 fit, $M_g/M_*$ (star-forming)')
ax.set_ylabel(r'log $M_{\rm cold}/M_*$, all galaxies'); ax.set_ylim(-3, 1); ax.legend(loc='lower left')
ax.set_title('(d) total cold gas', fontsize=10)
ax = axs[0, 2]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['H2gas'] / d['H1gas']), d['lm'] > 9, col, lab)
c18r = np.array([[10.22, -0.704], [10.59, -0.662], [10.91, -0.530]])
ax.plot(c18r[:, 0], c18r[:, 1], 'D', color=OBS, mfc='white', label=r'xGASS $\langle\log M_{\rm H_2}/M_{\rm HI}\rangle$ (Catinella+18 T3, no He)')
ax.set_ylabel(r'log $M_{\rm H_2}/M_{\rm HI}$ (H only), all galaxies'); ax.set_ylim(-2.5, 1); ax.legend(loc='lower left')
ax.set_title('(e) molecular-to-atomic ratio', fontsize=10)
ax = axs[1, 2]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['H1gas'] / d['sfr']), d['ms'] & (d['lm'] > 9) & (d['sfr'] > 0), col, lab)
ax.axhspan(9.65 - 0.44, 9.65 + 0.44, color=OBS, alpha=0.08, lw=0, label=r'Saintonge+17 MS $\langle\log t_{\rm dep}({\rm HI})\rangle$ = 9.65 ± 0.44')
ax.set_ylabel(r'log $t_{\rm dep}({\rm HI})$ [yr], MS galaxies'); ax.set_ylim(8.5, 11); ax.legend(loc='upper left')
ax.set_title('(f) atomic gas depletion time', fontsize=10)
for a in axs.flat:
    a.set_xlabel(r'log $M_*$ [M$_\odot$]'); a.set_xlim(9, 11.8)
fig.suptitle('Check 2: gas content at z = 0 (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_2_gas_z0.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ 3. population at z = 0
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
ax = axs[0, 0]
bw = 0.1; e = np.arange(8, 12.4, bw); c = e[:-1] + bw / 2
for run, col, lab in RUNS:
    d = run.get(s0)
    for sel, ls, tag in ((d['lm'] > 0, '-', 'all'), (d['sf'], '--', 'SF'), (~d['sf'] & (d['lm'] > 0), ':', 'quiescent')):
        n, _ = np.histogram(d['lm'][sel], bins=e)
        ax.plot(c, lg(n / run.vol / bw), color=col, ls=ls, label=f'{lab}, {tag}' if tag == 'all' else None)
gm = np.genfromtxt(DATA + 'smf/gama_smf_morph.ecsv', comments='#', skip_header=1)
tot = np.log10(np.nansum(10 ** gm[:, [1, 3, 5, 7]], axis=1))
ax.plot(gm[:, 0], tot, 'o', color=OBS, ms=5, label='GAMA morphological SMF, summed (h=0.7)')
ax.plot([], [], color='#52514e', ls='--', label='SF (sSFR > 1e-11)'); ax.plot([], [], color='#52514e', ls=':', label='quiescent')
ax.set_ylabel(r'log $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]'); ax.set_ylim(-5.5, -1); ax.set_xlim(8.5, 12.2); ax.legend(loc='lower left')
ax.set_title('(a) stellar mass function', fontsize=10)
ax = axs[0, 1]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['ssfr']), d['sf'] & (d['lm'] > 8.5), col, lab)
ax.plot(lm_, np.log10(speagle_ms_ssfr(lm_, 0.0)), color=OBS, ls=':', lw=1.5, label='Speagle+14 MS (z = 0)')
ax.plot(lm_, -0.344 * (lm_ - 9) - 9.822, color=OBS, ls='--', lw=1.5, label='xGASS SFMS (Catinella+18)')
ax.set_ylabel(r'log sSFR [yr$^{-1}$], star-forming'); ax.set_ylim(-11.2, -9); ax.legend(loc='lower left')
ax.set_title('(b) star-forming sequence', fontsize=10); ax.set_xlim(8.5, 12)
ax = axs[1, 0]
for run, col, lab in RUNS:
    d = run.get(s0); c_, o = binned(d['lm'], (~d['sf']).astype(float), EDGES, d['lm'] > 8.5, stat=np.mean)
    ax.plot(c_, o[:, 0], color=col, label=lab)
ax.set_ylabel('quiescent fraction (sSFR < 1e-11 /yr)'); ax.set_ylim(0, 1); ax.legend(loc='upper left')
ax.set_title('(c) quiescent fraction', fontsize=10); ax.set_xlim(8.5, 12)
ax = axs[1, 1]
for run, col, lab in RUNS:
    d = run.get(s0); m = (d['BulgeMass'] > 1e8) & (d['BlackHoleMass'] > 0)
    med(ax, lg(d['BulgeMass']), lg(d['BlackHoleMass']), m, col, lab)
mb = np.linspace(8.5, 12, 20)
ax.plot(mb, 8.69 + 1.16 * (mb - 11), color=OBS, ls='--', lw=1.5, label='Kormendy & Ho 13 (their eq. 10)')
ax.set_xlabel(r'log $M_{\rm bulge}$ [M$_\odot$]'); ax.set_ylabel(r'log $M_{\rm BH}$ [M$_\odot$]'); ax.set_xlim(8.5, 12)
ax.set_ylim(4, 10.5); ax.legend(loc='upper left'); ax.set_title('(d) black hole–bulge mass', fontsize=10)
for a in axs.flat[:3]:
    a.set_xlabel(r'log $M_*$ [M$_\odot$]')
fig.suptitle('Check 3: galaxy population at z = 0 (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_3_population_z0.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ 4. metals
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
ax = axs[0, 0]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], lg(d['MetalsStellarMass'] / d['StellarMass'] / 0.02), d['lm'] > 9, col, lab)
ax.plot(G05[:, 0], G05[:, 1], 'o', color=OBS, mfc='white', ms=5, label='Gallazzi+05, light-weighted')
ax.plot(G05[:, 0], 1.08 * G05[:, 1] - 0.16, 'o', color=OBS, ms=5, label='Gallazzi+05 → mass-weighted (Peeples+14 Eq. 7)')
ax.set_ylabel(r'log $Z_*/0.02$'); ax.set_ylim(-1.2, 0.5); ax.legend(loc='lower right')
ax.set_title('(a) stellar mass–metallicity, z = 0', fontsize=10)
oh = lambda z: 8.69 + lg(z / 0.0134)
ax = axs[0, 1]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], oh(d['MetalsColdGas'] / d['ColdGas']), d['sf'] & (d['lm'] > 9) & (d['ColdGas'] > 0), col, lab)
m = np.linspace(9, 11.5, 60)
ax.plot(m, 8.82 - 0.28 / 3.43 * np.log10(1 + (10 ** m / 10 ** 10.16) ** -3.43), color=OBS, lw=5, alpha=0.25, label='Sanders+21 z ~ 0')
C = np.loadtxt(DATA + 'metallicity/Curti2020.dat'); ax.plot(C[:, 0] - 0.04, C[:, 1], color=OBS, ls='--', lw=1.2, label='Curti+20')
ax.set_ylabel('12 + log(O/H) (Asplund+09 solar)'); ax.set_ylim(8.0, 9.3); ax.legend(loc='lower right')
ax.set_title('(b) gas-phase MZR, z = 0, star-forming', fontsize=10)
ax = axs[1, 0]
s2 = on.snap_at(2.3)
for run, col, lab in RUNS:
    d = run.get(s2); med(ax, d['lm'], oh(d['MetalsColdGas'] / d['ColdGas']), d['sf'] & (d['lm'] > 9) & (d['ColdGas'] > 0), col, lab)
m = np.linspace(9, 10.5, 20)
ax.plot(m, 8.51 + 0.30 * (m - 10), color=OBS, lw=5, alpha=0.25, label='Sanders+21 z ~ 2.3 (9–10.5)')
ax.set_ylabel('12 + log(O/H)'); ax.set_ylim(7.8, 9.0); ax.legend(loc='lower right')
ax.set_title(f'(c) gas-phase MZR, z = {on.z[s2]:.2f}', fontsize=10)
ax = axs[1, 1]
zs = [0, 0.5, 1, 1.5, 2, 2.5, 3]
for run, col, lab in RUNS:
    vg, vs = [], []
    for z in zs:
        d = run.get(run.snap_at(z)); mm = d['sf'] & (np.abs(d['lm'] - 10) < 0.2) & (d['ColdGas'] > 0)
        vg.append(np.median(oh(d['MetalsColdGas'][mm] / d['ColdGas'][mm])))
    ax.plot(zs, vg, 'o-', color=col, label=f'{lab}')
ax.plot([0, 2.3, 3.3], [8.77, 8.51, 8.41], 's', color=OBS, ms=7, label='Sanders+21 at 10$^{10}$ M$_\\odot$')
ax.set_xlabel('z'); ax.set_ylabel('12 + log(O/H) at log M* = 10'); ax.legend(loc='lower left')
ax.set_title('(d) gas metallicity evolution at fixed mass', fontsize=10)
for a in axs.flat[:3]:
    a.set_xlabel(r'log $M_*$ [M$_\odot$]'); a.set_xlim(9, 11.8)
fig.suptitle('Check 4: metallicities (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_4_metals.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ 5. evolution
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
snaps = range(20, 64)
ax = axs[0, 0]
for run, col, lab in RUNS:
    zz, rho = [], []
    for s in snaps:
        d = run.get(s); zz.append(run.z[s]); rho.append(np.sum(d['sfr']) / run.vol)
    ax.plot(np.log10(1 + np.array(zz)), np.log10(rho), color=col, label=lab)
D = np.loadtxt(DATA + 'sfrd/Driver18_sfr.dat')
zc = 0.5 * (D[:, 0] + D[:, 1])
ax.errorbar(np.log10(1 + zc), D[:, 2], yerr=np.hypot(D[:, 3], D[:, 5]), fmt='o', color=OBS, ms=4, capsize=2, lw=1, label='Driver+18 (Chabrier, h = 0.7)')
ax.set_xlabel('log (1 + z)'); ax.set_ylabel(r'log $\rho_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]'); ax.legend(loc='lower left')
ax.set_title('(a) cosmic star-formation rate density', fontsize=10)
ax = axs[0, 1]
for run, col, lab in RUNS:
    zz, rho = [], []
    for s in snaps:
        d = run.get(s); zz.append(run.z[s]); rho.append(np.sum(d['StellarMass']) / run.vol)
    ax.plot(np.log10(1 + np.array(zz)), np.log10(rho), color=col, label=lab)
ax.set_xlabel('log (1 + z)'); ax.set_ylabel(r'log $\rho_*$ [M$_\odot$ Mpc$^{-3}$]'); ax.legend(loc='lower left')
ax.set_title('(b) stellar mass density', fontsize=10)
ax = axs[1, 0]
zs = [0, 0.5, 1, 1.5, 2, 3, 4]
for run, col, lab in RUNS:
    v = []
    for z in zs:
        d = run.get(run.snap_at(z)); mm = d['ms'] & (np.abs(d['lm'] - 10.25) < 0.25)
        v.append(np.log10(np.mean(HE * d['H2gas'][mm] / d['StellarMass'][mm])) if mm.sum() > 10 else np.nan)
    ax.plot(np.log10(1 + np.array(zs)), v, 'o-', color=col, label=lab)
zz = np.linspace(0, 4, 40)
ax.plot(np.log10(1 + zz), -1.19 + 2.49 * np.log10(1 + zz) - 0.36 * (10.25 - 10.7), color=OBS, ls=':', lw=1.5, label=r'Tacconi+18 $\mu_{\rm gas}$ ($\beta$ = 0 form), MS')
ax.set_xlabel('log (1 + z)'); ax.set_ylabel(r'log $\langle M_{\rm H_2+He}/M_*\rangle$, MS, log M* = 10–10.5'); ax.legend(loc='upper left')
ax.set_title('(c) molecular gas fraction with redshift', fontsize=10)
ax = axs[1, 1]
for zt, ls in ((0, '-'), (1, '--'), (2, ':')):
    for run, col, lab in RUNS:
        d = run.get(run.snap_at(zt)); n, _ = np.histogram(d['lm'][d['lm'] > 0], bins=e)
        ax.plot(c, lg(n / run.vol / bw), color=col, ls=ls, label=f'{lab}' if zt == 0 else None)
    ax.plot([], [], color='#52514e', ls=ls, label=f'z = {zt}')
ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_ylabel(r'log $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]'); ax.set_ylim(-5.5, -1); ax.set_xlim(8.5, 12.2)
ax.legend(loc='lower left'); ax.set_title('(d) stellar mass function evolution', fontsize=10)
fig.suptitle('Check 5: evolution (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_5_evolution.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ 6. bursts and bulges
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
ax = axs[0, 0]
for run, col, lab in RUNS:
    zz, fb = [], []
    for s in snaps:
        d = run.get(s); zz.append(run.z[s]); fb.append(np.sum(d['SfrBulge']) / max(np.sum(d['sfr']), 1e-30))
    ax.plot(np.log10(1 + np.array(zz)), fb, color=col, label=lab)
ax.set_xlabel('log (1 + z)'); ax.set_ylabel('fraction of cosmic SFR in bursts (SfrBulge)'); ax.set_ylim(0, None); ax.legend(loc='upper left')
ax.set_title('(a) burst share of star formation', fontsize=10)
ax = axs[0, 1]
for run, col, lab in RUNS:
    d = run.get(s0); med(ax, d['lm'], d['BulgeMass'] / d['StellarMass'], d['lm'] > 9, col, lab)
ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_ylabel('B/T'); ax.set_ylim(0, 1); ax.set_xlim(9, 12); ax.legend(loc='upper left')
ax.set_title('(b) bulge-to-total ratio, z = 0', fontsize=10)
ax = axs[1, 0]
for run, col, lab in RUNS:
    d = run.get(s0)
    c_, o = binned(d['lm'], d['MergerBulgeMass'] / d['StellarMass'], EDGES, d['lm'] > 9, stat=np.mean); ax.plot(c_, o[:, 0], color=col, label=f'{lab}: merger bulge')
    c_, o = binned(d['lm'], d['InstabilityBulgeMass'] / d['StellarMass'], EDGES, d['lm'] > 9, stat=np.mean); ax.plot(c_, o[:, 0], color=col, ls='--', label=f'{lab}: instability bulge')
ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_ylabel('mean mass fraction of M*'); ax.set_xlim(9, 12); ax.set_ylim(0, 1); ax.legend(loc='upper left')
ax.set_title('(c) bulge origin, z = 0', fontsize=10)
ax = axs[1, 1]
for run, col, lab in RUNS:
    d = run.get(s0); n, _ = np.histogram(lg(d['BulgeMass'][d['BulgeMass'] > 0]), bins=e)
    ax.plot(c, lg(n / run.vol / bw), color=col, label=lab)
    sph = d['BulgeMass'] / np.maximum(d['StellarMass'], 1) > 0.5; n, _ = np.histogram(d['lm'][sph & (d['lm'] > 0)], bins=e)
    ax.plot(c, lg(n / run.vol / bw), color=col, ls='--')
ax.plot(gm[:, 0], gm[:, 1], 's', color=OBS, ms=5, label='GAMA E/HE (morphological SMF)')
ax.plot([], [], color='#52514e', ls='--', label='SAGE26 galaxies with B/T > 0.5')
ax.set_xlabel(r'log $M$ [M$_\odot$]'); ax.set_ylabel(r'log $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]'); ax.set_ylim(-5.5, -1.5); ax.set_xlim(8.5, 12.2)
ax.legend(loc='lower left'); ax.set_title('(d) bulge mass function (solid) and spheroid-dominated galaxies', fontsize=10)
fig.suptitle('Check 6: bursts and bulges (Millennium)', fontsize=12)
fig.tight_layout(); p = os.path.join(PLOTS, f'{PREFIX}_6_bursts_bulges.png'); fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)

# ------------------------------------------------------------------ numbers
for lo in (9.5, 10.0, 10.5, 11.0):
    row = []
    for run, col, lab in RUNS:
        d = run.get(s0); mm = (np.abs(d['lm'] - lo) < 0.1)
        ms_ = mm & d['ms']
        row.append(f"{lab[:22]:22s} H2+He/M*(MS) {np.log10(np.mean(HE * d['H2gas'][ms_] / d['StellarMass'][ms_])):+.2f} "
                   f"HI/M* {np.nanmedian(lg(d['H1gas'][mm] / d['StellarMass'][mm])):+.2f} H2/HI {np.nanmedian(lg(d['H2gas'][mm] / d['H1gas'][mm])):+.2f} "
                   f"sSFR(SF) {np.nanmedian(lg(d['ssfr'][mm & d['sf']])):+.2f} fQ {np.mean(~d['sf'][mm]):.2f} "
                   f"Z* {np.nanmedian(lg(d['MetalsStellarMass'][mm] / d['StellarMass'][mm] / 0.02)):+.2f} "
                   f"O/H {np.nanmedian(oh(d['MetalsColdGas'][mm & d['sf']] / d['ColdGas'][mm & d['sf']])):.2f}")
    print(f'logM* {lo}:\n  ' + '\n  '.join(row))
for run, col, lab in RUNS:
    d = run.get(s0)
    print(lab, 'rho_SFR z=0', np.log10(np.sum(d['sfr']) / run.vol).round(3), ' rho_* z=0', np.log10(np.sum(d['StellarMass']) / run.vol).round(3))

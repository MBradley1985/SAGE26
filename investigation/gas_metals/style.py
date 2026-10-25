"""Shared style, observational data paths and plotting helpers for figures 1-4."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, binned

DATA = '/Users/mbradley/Documents/PhD/SAGE26/data/'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
OBS = '#0b0b0b'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'legend.fontsize': 8, 'lines.linewidth': 2})
EDGES = np.arange(8.9, 12.0, 0.2)
ZSUN_G05 = 0.02          # Gallazzi+05 normalise to Z_sun = 0.02 (BC03); SAGE26 Z is mass fraction
# Gallazzi+05 Table 2: log M*, median, p16, p84 of light-weighted log Z*/Z_sun
G05 = np.array([[8.91,-0.60,-1.11,-0.00],[9.11,-0.61,-1.07,-0.00],[9.31,-0.65,-1.10,-0.05],[9.51,-0.61,-1.03,-0.01],
                [9.72,-0.52,-0.97,0.05],[9.91,-0.41,-0.90,0.09],[10.11,-0.23,-0.80,0.14],[10.31,-0.11,-0.65,0.17],
                [10.51,-0.01,-0.41,0.20],[10.72,0.04,-0.24,0.22],[10.91,0.07,-0.14,0.24],[11.11,0.10,-0.09,0.25],
                [11.31,0.12,-0.06,0.26],[11.51,0.13,-0.04,0.28],[11.72,0.14,-0.03,0.29],[11.91,0.15,-0.03,0.30]])


def band(ax, x, y, sel, col, lab, edges=EDGES):
    c, o = binned(x, y, edges, sel)
    ax.plot(c, o[:, 0], color=col, label=lab)
    ax.fill_between(c, o[:, 1], o[:, 2], color=col, alpha=0.15, lw=0)


def obs_lagos(ax, fname, lab, marker='o'):
    d = np.loadtxt(DATA + 'Gas/' + fname)
    ax.errorbar(d[:, 0], d[:, 1], yerr=[d[:, 2], d[:, 3]], fmt=marker, color=OBS, ms=5, capsize=2, lw=1, label=lab)


def lg(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)



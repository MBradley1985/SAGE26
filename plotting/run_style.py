#!/usr/bin/env python3
"""
run_style.py
============
Shared per-run visual styling for multi-run comparison plots (see
compare_runs.py). Run 0 (the "base" run) keeps the exact solid/full-color
look every single-run plot already uses; each additional run is drawn with
a different linestyle (cycling through a small set of dash patterns) and
its category colors lightened toward white, so a channel/category stays
recognizable across runs without two full-strength series looking identical
where they overlap.
"""

import matplotlib.colors as mcolors

LINESTYLES = ['-', '--', ':', '-.']
LIGHTEN_STEP = 0.35
MAX_LIGHTEN = 0.85

# Used only for scatter+KDE comparison panels (bh_lrd_analysis.py panels a-e
# and their bh_lrd_analysis_multiz.py equivalents), where linestyle is
# already spoken for (it encodes the 68/95/99.7% contour levels within one
# run) -- so runs are told apart by contour COLOR instead. Chosen to avoid
# the LRD red/orange (#C62828/#F57C00) and reference-line colors already
# used on those panels. Index 0 ('#333333', dark grey) matches the color
# every single-run background contour already uses.
CONTOUR_COLORS = ['#333333', '#1976D2', '#00897B', '#8E24AA']


def lighten_color(color, amount):
    """
    Blend any matplotlib color spec (hex string, named color, or RGB(A)
    tuple/array) toward white by `amount` (0 = unchanged, 1 = white).
    Always returns an (r, g, b) tuple in 0-1 floats.
    """
    if amount <= 0:
        return mcolors.to_rgb(color)
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount)


def style_for_index(i, label=None):
    """
    Style dict for the i-th run (0-based) in a comparison:
        label       run label for legends
        linestyle   '-' for the base run, cycling through dashes after that
        lighten     0.0 for the base run, increasing for each additional run
        show_band   whether to draw shaded uncertainty/percentile bands
                    (only the base run does, so two bands don't overlap
                    into an unreadable smear)
    """
    return {
        'label': label,
        'linestyle': LINESTYLES[min(i, len(LINESTYLES) - 1)],
        'lighten': 0.0 if i == 0 else min(MAX_LIGHTEN, LIGHTEN_STEP * i),
        'show_band': (i == 0),
    }


def contour_style_for_index(i, label=None):
    """Style dict for the i-th run's KDE-contour-only comparison panel
    (see CONTOUR_COLORS above): just a color and a label, no linestyle."""
    return {
        'label': label,
        'color': CONTOUR_COLORS[i % len(CONTOUR_COLORS)],
    }

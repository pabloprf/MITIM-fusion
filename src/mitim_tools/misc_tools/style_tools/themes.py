"""
mitim_tools.misc_tools.style_tools.themes
==========================================
Central place for MITIM-fusion figure styling.

Quick start
-----------
    from mitim_tools.misc_tools.style_tools.themes import apply_theme
    apply_theme()                      # interactive default
    apply_theme("paper")               # publication quality
    apply_theme("paper", latex=True)   # full LaTeX rendering
    apply_theme("dark")                # dark background

Font sizes scale automatically with the number of subplots:
    plt.subplots(1, 1)   →  base size  (e.g. 11 pt)
    plt.subplots(2, 3)   →  ~8 pt
    plt.subplots(4, 4)   →  ~7 pt
"""

from __future__ import annotations

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ── style file location ───────────────────────────────────────────────────────
_STYLES_DIR = pathlib.Path(__file__).parent / "styles"

_PRESETS = ("default", "paper", "dark")

# Base font size stored at apply_theme() time; read by _patched_subplots.
_base_size: float = 11.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_theme(
    preset: str = "default",
    *,
    size: float | None = None,
    latex: bool = False,
    dense_axes: bool = True,
    tight_layout: bool = True,
    colors: list[str] | None = None,
):
    """
    Apply a MITIM figure theme globally for the current session.

    Parameters
    ----------
    preset : {"default", "paper", "dark"}
        Named theme to load.
    size : float, optional
        Override the base font size for the preset. If not given, the
        preset's default is used (11 pt for default, 13 pt for paper).
        Font sizes then scale automatically with the number of subplots.
    latex : bool
        Enable full LaTeX text rendering (slow; requires a LaTeX install).
    dense_axes : bool
        Automatically call addDenseAxis() on every new Axes.
    tight_layout : bool
        Automatically enable tight layout on every new Figure.
    colors : list of str, optional
        Override the color cycle. Defaults to the MITIM Wong palette.
    """
    global _base_size

    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {_PRESETS}.")

    styles = [
        str(_STYLES_DIR / "mitim_base.mplstyle"),
        str(_STYLES_DIR / f"mitim_{preset}.mplstyle"),
    ]

    if preset == "dark":
        plt.style.use(["dark_background"] + styles)
    else:
        plt.style.use(styles)

    if size is not None:
        _base_size = float(size)
        _apply_size_rcparams(_base_size)
    else:
        # Read the base size that the mplstyle just set
        _base_size = float(plt.rcParams["font.size"])

    if latex:
        plt.rcParams["text.usetex"] = True

    if colors is not None:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)

    if dense_axes or tight_layout:
        _patch_pyplot(dense_axes=dense_axes, tight_layout=tight_layout)


def reset_theme():
    """Restore matplotlib defaults and remove any MITIM patches."""
    plt.rcdefaults()
    _unpatch_pyplot()


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic font scaling
# ─────────────────────────────────────────────────────────────────────────────

def _scaled_size(n_axes: int) -> float:
    """
    Return a font size scaled for a figure with n_axes subplots.

    Uses a gentle power-law decay clamped at 60 % of the base size:
        1 panel  → 100 %  (base)
        2 panels →  87 %
        4 panels →  76 %
        6 panels →  70 %
        9 panels →  63 %
       16 panels →  56 %  (clamped at 60 %)
    """
    scale = max(0.60, 1.0 / (max(1, n_axes) ** 0.20))
    return round(_base_size * scale, 1)


def _apply_size_rcparams(size: float) -> None:
    """Push a complete set of size-related rcParams."""
    plt.rcParams["font.size"]        = size
    plt.rcParams["axes.labelsize"]   = size
    plt.rcParams["axes.titlesize"]   = size
    plt.rcParams["axes.titlepad"]    = max(4, size * 0.6)
    plt.rcParams["xtick.labelsize"]  = size * 0.87
    plt.rcParams["ytick.labelsize"]  = size * 0.87
    plt.rcParams["legend.fontsize"]  = size * 0.87


# ─────────────────────────────────────────────────────────────────────────────
# plt.subplots / Figure.add_subplot patching
# ─────────────────────────────────────────────────────────────────────────────

_orig_subplots    = None
_orig_add_subplot = None


def _patch_pyplot(dense_axes: bool, tight_layout: bool):
    global _orig_subplots, _orig_add_subplot
    if _orig_subplots is not None:
        _unpatch_pyplot()

    from matplotlib.figure import Figure as _Figure

    _orig_subplots    = plt.subplots
    _orig_add_subplot = _Figure.add_subplot

    def _patched_subplots(*args, **kwargs):
        # ── dynamic font scaling ──────────────────────────────────────────────
        nrows = int(args[0]) if len(args) > 0 else int(kwargs.get("nrows", 1))
        ncols = int(args[1]) if len(args) > 1 else int(kwargs.get("ncols", 1))
        _apply_size_rcparams(_scaled_size(nrows * ncols))

        fig, axes = _orig_subplots(*args, **kwargs)

        if tight_layout:
            try:
                fig.set_layout_engine("tight")
            except AttributeError:
                fig.set_tight_layout(True)
        # dense_axes applied inside _patched_add_subplot (called by subplots)
        return fig, axes

    def _patched_add_subplot(self, *args, **kwargs):
        ax = _orig_add_subplot(self, *args, **kwargs)
        if dense_axes:
            _apply_dense_axes(ax)
        return ax

    plt.subplots      = _patched_subplots
    _Figure.add_subplot = _patched_add_subplot


def _unpatch_pyplot():
    global _orig_subplots, _orig_add_subplot
    from matplotlib.figure import Figure as _Figure
    if _orig_subplots is not None:
        plt.subplots        = _orig_subplots
        _orig_subplots      = None
    if _orig_add_subplot is not None:
        _Figure.add_subplot = _orig_add_subplot
        _orig_add_subplot   = None


def _apply_dense_axes(axes):
    from mitim_tools.misc_tools.GRAPHICStools import addDenseAxis
    if axes is None:
        return
    ax_array = np.asarray(axes).flatten()
    for ax in ax_array:
        try:
            addDenseAxis(ax)
        except Exception:
            pass  # e.g. colorbar axes — skip silently


# ─────────────────────────────────────────────────────────────────────────────
# MITIM color palette — Wong (2011), colorblind-safe, Nature/Science standard
# ─────────────────────────────────────────────────────────────────────────────

COLORS: list[str] = [
    # ── Wong (2011), colorblind-safe — yellow moved to end ────────────────────
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#999999",  # grey
    "#44AA99",  # teal
    "#F0E442",  # yellow (low contrast on white — kept last among Wong)
    # ── Legacy MITIM overflow (for plots with > 10 lines) ─────────────────────
    "b",
    "r",
    "m",
    "orange",
    "c",
    "g",
    "chocolate",
    "olive",
    "fuchsia",
    "slategrey",
]


def get_colors() -> list[str]:
    """Return the MITIM color list (Wong 2011 + legacy overflow)."""
    return list(COLORS)

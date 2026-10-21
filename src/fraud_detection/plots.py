"""Shared matplotlib styling so every notebook chart reads as one system."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-validated categorical slots (legit = slot 1, fraud = slot 2).
COLOR_LEGIT: str = "#2a78d6"
COLOR_FRAUD: str = "#eb6834"
COLOR_NEUTRAL: str = "#898781"
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
GRID: str = "#e1e0d9"
SURFACE: str = "#fcfcfb"


def set_style() -> None:
    """Apply a light, low-chrome style: recessive grid/axes, thin marks, readable text."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": COLOR_NEUTRAL,
            "ytick.color": COLOR_NEUTRAL,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 100,
            "figure.figsize": (10, 4),
        }
    )


def fraud_legit_colors() -> dict[int, str]:
    """Return the color mapping keyed by the `isFraud` value."""
    return {0: COLOR_LEGIT, 1: COLOR_FRAUD}


def finish(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> plt.Axes:
    """Set title and axis labels in one call.

    Args:
        ax: Axes to decorate.
        title: Chart title (states the takeaway when possible).
        xlabel: X axis label.
        ylabel: Y axis label.

    Returns:
        The same axes, for chaining.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

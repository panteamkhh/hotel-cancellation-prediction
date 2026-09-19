"""
Shared chart theme so every figure in the project looks the same.

``visualization.py`` and ``evaluation.py`` call :func:`apply_theme` once at
import time instead of each setting their own seaborn style, which keeps
the notebooks and the CLI-generated figures visually consistent.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR

# Categorical palettes used across the report, by chart family.
PALETTE_CATEGORICAL = "viridis"
PALETTE_CANCELLATION = "rocket"
PALETTE_REVENUE = "crest"
PALETTE_COUNTRY = "mako"
PALETTE_HOTEL = "Set2"
PALETTE_STAY = "flare"
CANCEL_COLORS = ["#4C72B0", "#C44E52"]

DPI = 150
FIGURE_DPI = DPI


def apply_theme() -> None:
    """Apply the shared seaborn/matplotlib theme globally."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def save_figure(fig, name: str, dpi: int = DPI):
    """Tighten, save a figure into ``reports/figures/`` and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    output_path = FIGURES_DIR / name
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path

"""
Shared style configuration for all paper figures.

This module provides a canonical color palette and style settings
to ensure visual consistency across all figures in the paper.

Usage:
    from paper_style import COLORS, setup_style
    setup_style()  # Call once at the start of your script
"""

import matplotlib.pyplot as plt


def setup_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "font.family": "serif",
        }
    )


# Canonical color palette for all paper figures
COLORS = {
    # Zero-intelligence hierarchy (Gray/Blue/Green scheme)
    # These MUST be consistent across all figures
    "zi": "#9E9E9E",  # Gray - unconstrained/chaotic baseline
    "zic": "#2196F3",  # Blue - budget constrained
    "zip": "#4CAF50",  # Green - adaptive learning
    # Market structure colors
    "demand": "#2196F3",  # Blue (buyer side)
    "supply": "#F44336",  # Red (seller side)
    "equilibrium": "#4CAF50",  # Green
    "surplus": "#81C784",  # Light green (realized gains)
    "v_ineff": "#FFB74D",  # Orange (V-inefficiency: missed trades)
    "em_ineff": "#E57373",  # Light red (EM-inefficiency: bad trades)
    # Buyer/seller distinction
    "buyer": "#2196F3",  # Blue
    "seller": "#F44336",  # Red
    "marginal": "#9E9E9E",  # Gray
    # Santa Fe tournament traders
    "kaplan": "#F44336",  # Red (aggressive sniper)
    "gd": "#4CAF50",  # Green (Gjerstad-Dickhaut)
    "ringuette": "#FF9800",  # Orange
    "skeleton": "#9C27B0",  # Purple
    "el": "#00BCD4",  # Cyan (Easley-Ledyard)
    "markup": "#795548",  # Brown
    # RL agents
    "ppo": "#e74c3c",  # Red (highlight color for PPO)
    # Neutral colors
    "gray": "#7f8c8d",  # Neutral gray for non-highlighted items
    "light_gray": "#95a5a6",  # Lighter gray
    # Legacy compatibility aliases
    "ZI": "#9E9E9E",
    "ZIC": "#2196F3",
    "ZIP": "#4CAF50",
    "GD": "#4CAF50",
    "Kaplan": "#F44336",
    "Ringuette": "#FF9800",
    "Skeleton": "#9C27B0",
    "Ledyard": "#00BCD4",
    "Markup": "#795548",
}


# Strategy ordering for consistent legends/charts
STRATEGY_ORDER = [
    "ZI",
    "ZIC",
    "ZIP",
    "GD",
    "Kaplan",
    "Ringuette",
    "Skeleton",
    "Ledyard",
    "Markup",
    "PPO",
]

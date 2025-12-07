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
            # Aesthetic improvements: cleaner spines and lighter grids
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
        }
    )


# Canonical color palette for all paper figures
# Deeper, more saturated colors for better visual impact
COLORS = {
    # Zero-intelligence hierarchy (Gray/Blue/Green scheme)
    # These MUST be consistent across all figures
    "zi": "#757575",  # Darker gray - unconstrained/chaotic baseline
    "zic": "#1976D2",  # Deeper blue - budget constrained
    "zip": "#388E3C",  # Deeper green - adaptive learning
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
    # Santa Fe tournament traders (12 original traders)
    "kaplan": "#F44336",  # Red (aggressive sniper)
    "gd": "#4CAF50",  # Green (Gjerstad-Dickhaut)
    "ringuette": "#FF9800",  # Orange
    "skeleton": "#9C27B0",  # Purple
    "el": "#00BCD4",  # Cyan (Easley-Ledyard)
    "ledyard": "#00BCD4",  # Cyan (alias for EL)
    "perry": "#E91E63",  # Pink (adaptive parameters)
    "breton": "#3F51B5",  # Indigo (fixed-rule)
    "bgan": "#009688",  # Teal (belief-based)
    "jacobson": "#607D8B",  # Blue gray (equilibrium estimation)
    "lin": "#8BC34A",  # Light green (statistical)
    "staecker": "#FF5722",  # Deep orange
    "gamer": "#673AB7",  # Deep purple
    "markup": "#795548",  # Brown
    # RL agents
    "ppo": "#e74c3c",  # Red (highlight color for PPO)
    # Neutral colors
    "gray": "#7f8c8d",  # Neutral gray for non-highlighted items
    "light_gray": "#95a5a6",  # Lighter gray
    # ZI2 (market-aware but not learning)
    "zi2": "#F57C00",  # Deeper orange - market-aware, between ZIC and ZIP
    # Numbered variants (ZIC1/ZIC2, ZIP1/ZIP2)
    "zic1": "#1976D2",  # Blue - budget constrained (same as zic)
    "zic2": "#F57C00",  # Orange - market-aware (same as zi2)
    "zip1": "#388E3C",  # Green - adaptive learning (same as zip)
    "zip2": "#2E7D32",  # Darker green - learning + market-aware
    # Legacy compatibility aliases (must match lowercase versions)
    "ZI": "#757575",
    "ZIC": "#1976D2",
    "ZI2": "#F57C00",
    "ZIP": "#388E3C",
    "ZIC1": "#1976D2",
    "ZIC2": "#F57C00",
    "ZIP1": "#388E3C",
    "ZIP2": "#2E7D32",
    "GD": "#4CAF50",
    "Kaplan": "#F44336",
    "Ringuette": "#FF9800",
    "Skeleton": "#9C27B0",
    "Ledyard": "#00BCD4",
    "EL": "#00BCD4",
    "Perry": "#E91E63",
    "Breton": "#3F51B5",
    "BGAN": "#009688",
    "Jacobson": "#607D8B",
    "Lin": "#8BC34A",
    "Staecker": "#FF5722",
    "Gamer": "#673AB7",
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
    "Perry",
    "Breton",
    "BGAN",
    "Jacobson",
    "Lin",
    "Staecker",
    "Gamer",
    "Markup",
    "PPO",
]

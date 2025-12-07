#!/usr/bin/env python3
"""
Generate Efficiency Decomposition Figure for Section 5.

Creates a waterfall/stacked bar chart showing efficiency progression:
ZI (27%) → ZIC (+64pp = 91%) → ZI2 (+4pp = 95%) → ZIP (+5pp = 100%)

This visualizes:
- Finding #1: Institutions create efficiency (ZI→ZIC is the big jump)
- Finding #4: Two-dimensional ladder (constraint axis vs learning axis)

Data source: paper/arxiv/figures/table_foundational.tex

Usage:
    python scripts/plot_efficiency_decomposition.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()


def main() -> None:
    """Generate the efficiency decomposition waterfall chart."""
    # Data from table_foundational.tex (BASE environment)
    # These are file-backed values
    strategies = ["ZI", "ZIC", "ZI2", "ZIP"]
    efficiencies = [27, 91, 95, 100]

    # Calculate incremental gains
    gains = [efficiencies[0]]  # ZI baseline
    for i in range(1, len(efficiencies)):
        gains.append(efficiencies[i] - efficiencies[i - 1])

    # Labels for the gains
    gain_labels = [
        "Baseline",
        "+64pp\n(Budget\nConstraint)",
        "+4pp\n(Market\nAwareness)",
        "+5pp\n(Adaptive\nLearning)",
    ]

    # Colors from paper_style
    colors = [COLORS["ZI"], COLORS["ZIC"], COLORS["ZI2"], COLORS["ZIP"]]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Waterfall chart
    x = np.arange(len(strategies))
    bar_width = 0.6

    # Track cumulative height for stacking
    cumulative = 0
    bars = []

    for i, (strategy, gain, color, label) in enumerate(zip(strategies, gains, colors, gain_labels)):
        # Draw bar from cumulative to cumulative + gain
        bar = ax.bar(
            x[i],
            gain,
            bar_width,
            bottom=cumulative,
            color=color,
            edgecolor="white",
            linewidth=2,
            label=f"{strategy}: {efficiencies[i]}%",
        )
        bars.append(bar)

        # Add gain label inside the bar
        bar_center = cumulative + gain / 2
        text_color = "white" if gain > 10 else "black"
        ax.text(
            x[i],
            bar_center,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

        # Add efficiency value at top of bar
        ax.text(
            x[i],
            cumulative + gain + 1.5,
            f"{efficiencies[i]}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

        cumulative += gain

    # Add connecting lines to show the waterfall flow
    for i in range(len(strategies) - 1):
        ax.plot(
            [x[i] + bar_width / 2, x[i + 1] - bar_width / 2],
            [efficiencies[i], efficiencies[i]],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )

    # Axis annotations for the two axes
    # Constraint axis (ZI → ZIC)
    ax.annotate(
        "",
        xy=(0.5, 60),
        xytext=(0.5, 27),
        arrowprops=dict(arrowstyle="->", color="#1976D2", lw=2),
    )
    ax.text(
        -0.3,
        45,
        "Constraint\nAxis\n(+64pp)",
        fontsize=10,
        ha="center",
        va="center",
        color="#1976D2",
        fontweight="bold",
    )

    # Learning axis (ZIC → ZIP)
    ax.annotate(
        "",
        xy=(2.5, 100),
        xytext=(2.5, 91),
        arrowprops=dict(arrowstyle="->", color="#388E3C", lw=2),
    )
    ax.text(
        3.3,
        95,
        "Learning\nAxis\n(+9pp)",
        fontsize=10,
        ha="center",
        va="center",
        color="#388E3C",
        fontweight="bold",
    )

    # Labels and styling
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=13, fontweight="bold")
    ax.set_ylabel("Allocative Efficiency (%)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, 3.6)

    # Add horizontal reference lines
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(3.5, 101, "Perfect Efficiency", ha="right", va="bottom", fontsize=9, color="gray")

    # Title
    ax.set_title(
        "Efficiency Decomposition: Institutions vs Intelligence",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Add insight annotation
    insight_text = (
        "Budget constraint alone achieves 87% of the\n"
        "efficiency gap (27%→91%). Learning adds\n"
        "the remaining 13% (91%→100%)."
    )
    ax.text(
        0.98,
        0.02,
        insight_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()

    # Save
    output_dir = Path("paper/arxiv/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "efficiency_decomposition.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved PNG to {png_path}")

    pdf_path = output_dir / "efficiency_decomposition.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF to {pdf_path}")

    plt.close()


if __name__ == "__main__":
    main()

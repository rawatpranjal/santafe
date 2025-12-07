#!/usr/bin/env python3
"""
Generate Robustness Heatmap Figure for Section 5.

Creates a heatmap showing efficiency across strategies and environments:
- Rows: ZI, ZIC, ZIP
- Columns: Standard (BASE, PER, LAD) | Stress (SHRT, RAN, SML) | Asymmetric (BBBS, BSSS)

This visualizes:
- Finding #6: Robustness under stress reveals the value of learning
- Finding #3: ZIP dominates under stress while ZIC collapses in SHRT

Data source: paper/arxiv/figures/table_efficiency_full.tex

Usage:
    python scripts/plot_robustness_heatmap.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, ".")

from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()


def main() -> None:
    """Generate the robustness heatmap."""
    # Data from table_efficiency_full.tex (file-backed values)
    # Format: efficiency values by strategy and environment
    strategies = ["ZI", "ZIC", "ZIP"]

    # Organized by category
    env_groups = {
        "Standard": ["BASE", "PER", "LAD"],
        "Stress": ["SHRT", "RAN", "SML"],
        "Asymmetric": ["BBBS", "BSSS"],
    }

    # Efficiency data from table_efficiency_full.tex
    data = {
        "ZI": {
            "BASE": 28,
            "BBBS": 55,
            "BSSS": 53,
            "EQL": 100,
            "RAN": 83,
            "PER": 28,
            "SHRT": 29,
            "TOK": 94,
            "SML": 16,
            "LAD": 28,
        },
        "ZIC": {
            "BASE": 97,
            "BBBS": 97,
            "BSSS": 97,
            "EQL": 100,
            "RAN": 100,
            "PER": 98,
            "SHRT": 78,
            "TOK": 96,
            "SML": 88,
            "LAD": 98,
        },
        "ZIP": {
            "BASE": 99,
            "BBBS": 99,
            "BSSS": 100,
            "EQL": 100,
            "RAN": 97,
            "PER": 100,
            "SHRT": 99,
            "TOK": 99,
            "SML": 89,
            "LAD": 99,
        },
    }

    # Flatten environments for the heatmap
    envs = []
    for group_envs in env_groups.values():
        envs.extend(group_envs)

    # Build data matrix
    matrix = np.zeros((len(strategies), len(envs)))
    for i, strat in enumerate(strategies):
        for j, env in enumerate(envs):
            matrix[i, j] = data[strat][env]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Custom colormap: red (low) -> yellow (mid) -> green (high)
    colors_cmap = ["#d32f2f", "#ffeb3b", "#388e3c"]
    cmap = LinearSegmentedColormap.from_list("efficiency", colors_cmap)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Efficiency (%)", fontsize=12)

    # Add text annotations in cells
    for i in range(len(strategies)):
        for j in range(len(envs)):
            value = matrix[i, j]
            # Use white text on dark backgrounds
            text_color = "white" if value < 60 else "black"
            ax.text(
                j,
                i,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    # Highlight stress environments with red border
    stress_start = len(env_groups["Standard"])
    stress_end = stress_start + len(env_groups["Stress"])
    rect = mpatches.Rectangle(
        (stress_start - 0.5, -0.5),
        len(env_groups["Stress"]),
        len(strategies),
        linewidth=3,
        edgecolor="#d32f2f",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    # Add "STRESS" label
    ax.text(
        stress_start + len(env_groups["Stress"]) / 2 - 0.5,
        -0.85,
        "STRESS",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#d32f2f",
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels(envs, fontsize=11)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=12, fontweight="bold")

    # Color strategy labels
    for i, strat in enumerate(strategies):
        color = COLORS.get(strat, "black")
        ax.get_yticklabels()[i].set_color(color)

    # Add group separators
    sep_positions = [
        len(env_groups["Standard"]) - 0.5,
        len(env_groups["Standard"]) + len(env_groups["Stress"]) - 0.5,
    ]
    for pos in sep_positions:
        ax.axvline(pos, color="gray", linewidth=2, alpha=0.5)

    # Add group headers at top
    group_starts = [
        0,
        len(env_groups["Standard"]),
        len(env_groups["Standard"]) + len(env_groups["Stress"]),
    ]
    group_widths = [len(v) for v in env_groups.values()]
    group_names = list(env_groups.keys())

    for start, width, name in zip(group_starts, group_widths, group_names):
        ax.text(
            start + width / 2 - 0.5,
            -1.3,
            name,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#555",
        )

    # Key insight annotation
    # Highlight the SHRT cell for ZIC (78%) vs ZIP (99%)
    shrt_col = envs.index("SHRT")
    zic_row = strategies.index("ZIC")
    ax.add_patch(mpatches.Circle((shrt_col, zic_row), 0.4, color="none", ec="white", lw=3))

    insight_text = "ZIC collapses under time pressure\n" "(SHRT: 78% vs ZIP's 99%)"
    ax.annotate(
        insight_text,
        xy=(shrt_col, zic_row),
        xytext=(shrt_col + 2, zic_row + 0.8),
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
        arrowprops=dict(arrowstyle="->", color="gray", connectionstyle="arc3,rad=0.2"),
    )

    # Title
    ax.set_title(
        "Robustness Across Market Environments",
        fontsize=14,
        fontweight="bold",
        pad=25,
    )

    plt.tight_layout()

    # Save
    output_dir = Path("paper/arxiv/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "robustness_heatmap.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved PNG to {png_path}")

    pdf_path = output_dir / "robustness_heatmap.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF to {pdf_path}")

    plt.close()


if __name__ == "__main__":
    main()

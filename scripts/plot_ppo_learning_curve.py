#!/usr/bin/env python3
"""
Plot PPO learning curve vs legacy strategy baselines.

Usage:
    python scripts/plot_ppo_learning_curve.py --input results/learning_curve.json --output figures/ppo_learning_curve.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()


def main():
    parser = argparse.ArgumentParser(description="Plot PPO learning curves")
    parser.add_argument("--input", required=True, help="Input JSON from eval_ppo_checkpoints.py")
    parser.add_argument("--output", default="figures/ppo_learning_curve.pdf", help="Output figure")
    parser.add_argument(
        "--title", default="PPO Learning Curve vs Legacy Strategies", help="Plot title"
    )
    args = parser.parse_args()

    # Load results
    with open(args.input) as f:
        data = json.load(f)

    # Extract PPO data points
    timesteps = []
    profits = []

    for cp in data["checkpoints"]:
        if cp["profit"] is not None:
            ts = cp["timesteps"]
            if ts == "final":
                # Use last timestep + 1M as approximation
                ts = max(timesteps) + 1_000_000 if timesteps else 10_000_000
            timesteps.append(ts)
            profits.append(cp["profit"])

    # Convert to arrays
    timesteps = np.array(timesteps) / 1e6  # Convert to millions
    profits = np.array(profits)

    # Baselines
    baselines = data["baselines"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PPO learning curve (use COLORS palette)
    ax.plot(
        timesteps,
        profits,
        "-o",
        color=COLORS["ppo"],
        linewidth=2,
        markersize=8,
        label="PPO",
        zorder=10,
    )

    # Plot horizontal lines for baselines (use COLORS palette)
    for name, profit in sorted(baselines.items(), key=lambda x: -x[1]):
        color = COLORS.get(name, COLORS["gray"])
        ax.axhline(y=profit, color=color, linestyle="--", alpha=0.7, label=name)
        ax.text(max(timesteps) * 1.02, profit, name, va="center", color=color)

    # Find crossover point with Ringuette
    ringuette = baselines["Ringuette"]
    crossover_idx = None
    for i, p in enumerate(profits):
        if p > ringuette:
            crossover_idx = i
            break

    if crossover_idx is not None:
        ax.axvline(x=timesteps[crossover_idx], color=COLORS["zip"], linestyle=":", alpha=0.8)
        ax.annotate(
            f"PPO beats Ringuette\n@ {timesteps[crossover_idx]:.1f}M steps",
            xy=(timesteps[crossover_idx], ringuette),
            xytext=(timesteps[crossover_idx] + 0.5, ringuette - 100),
            arrowprops=dict(arrowstyle="->", color=COLORS["zip"]),
            color=COLORS["zip"],
        )

    # Labels and formatting
    ax.set_xlabel("Training Steps (Millions)")
    ax.set_ylabel("Tournament Profit")
    ax.set_title(args.title)

    # Set y-axis to show full range
    min_profit = min(min(profits), min(baselines.values())) - 50
    max_profit = max(max(profits), max(baselines.values())) + 50
    ax.set_ylim(min_profit, max_profit)

    # Grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    # Also save PNG
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")

    # Print ASCII summary
    print("\n" + "=" * 60)
    print("LEARNING CURVE DATA")
    print("=" * 60)
    print(f"{'Steps (M)':<12} {'Profit':<10} {'vs Ringuette':<15} {'Rank'}")
    print("-" * 60)

    for cp in data["checkpoints"]:
        ts = cp["timesteps"]
        ts_str = f"{ts/1e6:.1f}" if isinstance(ts, (int, float)) else ts
        profit = cp["profit"]
        rank = cp["rank"]

        if profit is not None:
            diff = profit - ringuette
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            status = "WINNER!" if rank == 1 else f"#{rank}"
            print(f"{ts_str:<12} {profit:<10.1f} {diff_str:<15} {status}")


if __name__ == "__main__":
    main()

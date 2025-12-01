"""Generate PPO tournament figures for the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()

# Data from tournament_results_20251129_200757.json (8M checkpoint)
RESULTS_8M = {
    "PPO (8M)": {"mean_profit": 1404.2, "std_profit": 715.6, "rank": 1},
    "Ringuette": {"mean_profit": 1384.2, "std_profit": 585.8, "rank": 2},
    "Ledyard": {"mean_profit": 1251.5, "std_profit": 808.3, "rank": 3},
    "GD": {"mean_profit": 1184.6, "std_profit": 631.5, "rank": 4},
    "Markup": {"mean_profit": 1131.4, "std_profit": 607.1, "rank": 5},
    "Skeleton": {"mean_profit": 1124.7, "std_profit": 648.1, "rank": 6},
    "Kaplan": {"mean_profit": 1119.3, "std_profit": 688.7, "rank": 7},
    "ZIC": {"mean_profit": 891.3, "std_profit": 426.7, "rank": 8},
    "ZIP": {"mean_profit": 863.2, "std_profit": 534.5, "rank": 9},
}

# Learning curve data from training log eval rewards (steps -> reward)
# Extracted from: grep "Eval num_timesteps" logs/ppo_v10_10M.log
EVAL_REWARDS = [
    (0.4, 1549.6),
    (0.8, 1211.8),
    (1.2, 1117.0),
    (1.6, 1244.0),
    (2.0, 1517.1),
    (2.4, 1273.2),
    (2.8, 1152.3),
    (3.2, 1441.2),
    (3.6, 1692.9),
    (4.0, 1558.1),
    (4.4, 1339.8),
    (4.8, 1315.0),
    (5.2, 1435.1),
    (5.6, 843.3),
    (6.0, 815.8),
    (6.4, 1185.2),
    (6.8, 1561.3),
    (7.2, 1023.9),
    (7.6, 1369.0),
    (8.0, 1545.4),
    (8.4, 1541.4),
]

# Tournament profit at 8M checkpoint (different from eval reward)
TOURNAMENT_8M = 1404.2

# Legacy baselines for reference lines
LEGACY_BASELINES = {
    "Ringuette": 1384.2,
    "Ledyard": 1251.5,
    "Kaplan": 1119.3,
    "Skeleton": 1124.7,
    "GD": 1184.6,
    "ZIC": 891.3,
}


def generate_bar_chart(output_path: Path):
    """Generate bar chart comparing 9 strategies ranked by profit."""
    # Sort by profit descending
    sorted_strategies = sorted(RESULTS_8M.items(), key=lambda x: x[1]["mean_profit"], reverse=True)

    names = [s[0] for s in sorted_strategies]
    profits = [s[1]["mean_profit"] for s in sorted_strategies]
    stds = [s[1]["std_profit"] for s in sorted_strategies]

    # Colors: PPO in red, others in gray (use COLORS for consistency)
    colors = [COLORS["ppo"] if "PPO" in name else COLORS["gray"] for name in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        names, profits, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5
    )

    # Add value labels on bars
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        ax.annotate(
            f"{profit:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Mean Profit")
    ax.set_xlabel("Strategy")
    ax.set_title("Extended Tournament Results (9 Strategies, PPO @ 8M Training Steps)")

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha="right")

    # Add horizontal line at Ringuette level
    ax.axhline(
        y=RESULTS_8M["Ringuette"]["mean_profit"],
        color=COLORS["Ringuette"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Ringuette baseline",
    )

    ax.legend(loc="upper right")
    ax.set_ylim(0, max(profits) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart to {output_path}")


def generate_learning_curve(output_path: Path):
    """Generate learning curve showing PPO progress vs legacy baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PPO eval rewards from training log
    steps = [x[0] for x in EVAL_REWARDS]
    rewards = [x[1] for x in EVAL_REWARDS]

    # Plot raw data points
    ax.scatter(steps, rewards, color=COLORS["ppo"], s=40, alpha=0.6, zorder=4)

    # Add smoothed line (moving average)
    window = 3
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_steps = steps[window // 2 : -(window // 2)] if window > 1 else steps
    ax.plot(
        smoothed_steps,
        smoothed,
        "-",
        color=COLORS["ppo"],
        linewidth=2.5,
        label="PPO (eval reward)",
        zorder=5,
    )

    # Plot horizontal lines for legacy baselines (use COLORS palette)
    for name, profit in LEGACY_BASELINES.items():
        ax.axhline(
            y=profit, color=COLORS[name], linestyle="--", linewidth=1.5, alpha=0.7, label=name
        )

    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("PPO Learning Curve vs Legacy Strategies")

    # Set x-axis to show expected range
    ax.set_xlim(0, 10.5)
    ax.set_ylim(700, 1800)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend outside plot
    ax.legend(loc="upper right")

    # Add annotation showing PPO beats Ringuette at 8M
    ax.annotate(
        "PPO beats Ringuette\nat 8M steps",
        xy=(8.0, 1545.4),
        xytext=(6.5, 1750),
        ha="left",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curve to {output_path}")


def main():
    figures_dir = Path("/Users/pranjal/Code/santafe-1/paper/arxiv/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate both figures
    generate_bar_chart(figures_dir / "ppo_tournament_bar.pdf")
    generate_learning_curve(figures_dir / "ppo_learning_curve.pdf")

    print("\nDone! Figures saved to paper/arxiv/figures/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Santa Fe Evolutionary Dynamics Figure for Section 6.

Creates a line chart showing population shares over 50 generations,
aggregated across multiple seeds.

Key findings to visualize:
- Skeleton dominance (62.5% of final population)
- Kaplan/Ringuette persistence (~13%/10%)
- ZIP extinction by generation 7.5

Usage:
    python scripts/plot_santafe_evolutionary_dynamics.py
    python scripts/plot_santafe_evolutionary_dynamics.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()

RESULTS_DIR = Path("results")

# Key strategies to highlight (in display order)
KEY_STRATEGIES = ["Skeleton", "Kaplan", "Ringuette", "ZIP", "ZIC", "GD", "EL"]


def load_evolution_results(seed: int, version: str = "v3") -> dict | None:
    """Load evolutionary tournament results for a specific seed."""
    path = RESULTS_DIR / f"evolution_{version}_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_population_shares(data: dict) -> dict[str, list[float]]:
    """Extract population share time series for each strategy."""
    generations = data.get("generations", [])
    pop_size = data.get("config", {}).get("population_size", 32)

    shares: dict[str, list[float]] = defaultdict(list)

    for gen_data in generations:
        counts = gen_data.get("strategy_counts", {})
        for strategy in KEY_STRATEGIES:
            count = counts.get(strategy, 0)
            shares[strategy].append(100 * count / pop_size)

    return dict(shares)


def aggregate_across_seeds(
    all_shares: list[dict[str, list[float]]],
) -> dict[str, tuple[list[float], list[float]]]:
    """Aggregate population shares across seeds, returning mean and std."""
    if not all_shares:
        return {}

    # Find max generations
    max_gens = max(len(s.get("Skeleton", [])) for s in all_shares)

    aggregated = {}
    for strategy in KEY_STRATEGIES:
        # Collect all values for each generation
        gen_values = [[] for _ in range(max_gens)]

        for seed_shares in all_shares:
            strategy_shares = seed_shares.get(strategy, [])
            for g, val in enumerate(strategy_shares):
                if g < max_gens:
                    gen_values[g].append(val)

        # Calculate mean and std for each generation
        means = []
        stds = []
        for vals in gen_values:
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        aggregated[strategy] = (means, stds)

    return aggregated


def generate_figure(
    aggregated: dict[str, tuple[list[float], list[float]]], output_path: Path, num_seeds: int
) -> None:
    """Generate the evolutionary dynamics figure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Line styles for different strategy types
    line_styles = {
        "Skeleton": "-",  # Solid (dominant)
        "Kaplan": "--",  # Dashed (sniper)
        "Ringuette": "--",  # Dashed (sniper)
        "ZIP": ":",  # Dotted (extinct)
        "ZIC": "-.",  # Dash-dot (baseline)
        "GD": "-.",  # Dash-dot
        "EL": ":",  # Dotted
    }

    for strategy in KEY_STRATEGIES:
        if strategy not in aggregated:
            continue

        means, stds = aggregated[strategy]
        generations = list(range(len(means)))
        color = COLORS.get(strategy, "#757575")
        linestyle = line_styles.get(strategy, "-")

        # Plot mean line
        ax.plot(generations, means, label=strategy, color=color, linewidth=2.5, linestyle=linestyle)

        # Add shaded confidence band (1 std)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.fill_between(
            generations, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15
        )

    # Add annotations for key events
    # ZIP extinction (around generation 7-11)
    ax.annotate(
        "ZIP extinct\n(gen ~8)",
        xy=(10, 2),
        xytext=(18, 15),
        fontsize=9,
        color=COLORS.get("ZIP", "#388E3C"),
        arrowprops=dict(arrowstyle="->", color=COLORS.get("ZIP", "#388E3C"), lw=1.5),
    )

    # Skeleton dominance
    ax.annotate(
        "Skeleton\ndominance",
        xy=(45, 60),
        xytext=(35, 75),
        fontsize=9,
        color=COLORS.get("Skeleton", "#9C27B0"),
        arrowprops=dict(arrowstyle="->", color=COLORS.get("Skeleton", "#9C27B0"), lw=1.5),
    )

    # Styling
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Population Share (%)", fontsize=12)
    ax.set_title(
        f"Evolutionary Dynamics of Santa Fe Trading Strategies (n={num_seeds} seeds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)

    # Legend outside plot
    ax.legend(loc="upper right", fontsize=10, ncol=2)

    # Add horizontal reference lines
    ax.axhline(50, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    # Also save PDF
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF to {pdf_path}")

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Santa Fe Evolutionary Dynamics Figure")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seeds to aggregate (default: 0-9)",
    )
    parser.add_argument(
        "--version", type=str, default="v3", help="Evolution results version (default: v3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/santafe_evolutionary_dynamics.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    print(f"Generating evolutionary dynamics figure from {len(args.seeds)} seeds...")

    all_shares = []
    for seed in args.seeds:
        data = load_evolution_results(seed, args.version)
        if data is None:
            print(f"  Warning: Missing results for seed {seed}")
            continue

        shares = extract_population_shares(data)
        all_shares.append(shares)
        print(f"  Loaded seed {seed}: {len(shares)} strategies tracked")

    if not all_shares:
        print("Error: No valid evolution results found")
        return

    print(f"\nAggregating across {len(all_shares)} seeds...")
    aggregated = aggregate_across_seeds(all_shares)

    # Print final population statistics
    print("\nFinal population shares (generation 49):")
    for strategy in KEY_STRATEGIES:
        if strategy in aggregated:
            means, stds = aggregated[strategy]
            if means:
                final_mean = means[-1]
                final_std = stds[-1]
                print(f"  {strategy}: {final_mean:.1f}% +/- {final_std:.1f}%")

    generate_figure(aggregated, args.output, len(all_shares))


if __name__ == "__main__":
    main()

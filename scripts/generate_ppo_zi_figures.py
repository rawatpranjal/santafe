"""Generate PPO vs ZI analysis figures for the paper.

Creates:
1. Profit comparison bar chart (PPO > ZIP > ZIC > ZI)
2. Market efficiency comparison (ZI vs ZIC vs ZIP vs PPO+mix)
3. Price volatility comparison
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()


def load_results():
    """Load results from JSON file."""
    results_path = Path("results/ppo_vs_zi_metrics/full_results.json")
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run scripts/run_ppo_zi_metrics.py first.")
        return None

    with open(results_path) as f:
        return json.load(f)


def generate_profit_bar_chart(results: dict, output_path: Path):
    """Generate horizontal bar chart of profit by strategy in PPO+mix market."""
    ppo_mix = results["ppo_mix"]
    profits = ppo_mix["profits"]

    # Sort by profit descending
    sorted_strategies = sorted(profits.items(), key=lambda x: x[1]["mean"], reverse=True)

    names = [s[0] for s in sorted_strategies]
    means = [s[1]["mean"] for s in sorted_strategies]
    stds = [s[1]["std"] for s in sorted_strategies]

    # Colors: PPO in red, others in gray (use COLORS for consistency)
    colors = [COLORS["ppo"] if name == "PPO" else COLORS["gray"] for name in names]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(names))
    bars = ax.barh(
        y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel("Mean Profit (10 seeds, 50 rounds each)")
    ax.set_title("PPO vs Zero-Intelligence: Profit Comparison")

    # Add value labels
    for bar, mean in zip(bars, means):
        width = bar.get_width()
        ax.annotate(
            f"{mean:,.0f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved profit bar chart to {output_path}")


def generate_efficiency_comparison(results: dict, output_path: Path):
    """Generate grouped bar chart comparing market efficiency."""
    configs = ["zi_only", "zic_only", "zip_only", "ppo_mix"]
    labels = ["ZI Only", "ZIC Only", "ZIP Only", "PPO + Mix"]

    efficiencies = []
    stds = []

    for config in configs:
        m = results[config]["market_metrics"]["efficiency"]
        efficiencies.append(m["mean"])
        stds.append(m["std"])

    # Use COLORS for consistency: PPO in red, ZI variants in blue
    colors = [COLORS["ppo"] if "ppo" in c else COLORS["zic"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(
        x, efficiencies, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Allocative Efficiency (%)")
    ax.set_title("Market Efficiency by Composition")
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.annotate(
            f"{eff:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Add horizontal line at 100%
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved efficiency comparison to {output_path}")


def generate_volatility_comparison(results: dict, output_path: Path):
    """Generate bar chart comparing price volatility."""
    configs = ["zi_only", "zic_only", "zip_only", "ppo_mix"]
    labels = ["ZI Only", "ZIC Only", "ZIP Only", "PPO + Mix"]

    volatilities = []
    stds = []

    for config in configs:
        m = results[config]["market_metrics"]["price_volatility"]
        volatilities.append(m["mean"])
        stds.append(m["std"])

    # Use COLORS for consistency: PPO in red, ZI variants in green
    colors = [COLORS["ppo"] if "ppo" in c else COLORS["zip"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(
        x, volatilities, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Price Volatility (% of equilibrium)")
    ax.set_title("Price Volatility by Market Composition")

    # Add value labels
    for bar, vol in zip(bars, volatilities):
        height = bar.get_height()
        ax.annotate(
            f"{vol:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved volatility comparison to {output_path}")


def generate_combined_metrics(results: dict, output_path: Path):
    """Generate combined 2x2 subplot figure with all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = ["zi_only", "zic_only", "zip_only", "ppo_mix"]
    labels = ["ZI Only", "ZIC Only", "ZIP Only", "PPO+Mix"]

    # 1. Profit comparison (top left)
    ax = axes[0, 0]
    ppo_mix = results["ppo_mix"]
    profits = ppo_mix["profits"]
    sorted_strats = sorted(profits.items(), key=lambda x: x[1]["mean"], reverse=True)
    names = [s[0] for s in sorted_strats]
    means = [s[1]["mean"] for s in sorted_strats]
    stds = [s[1]["std"] for s in sorted_strats]
    colors = [COLORS["ppo"] if n == "PPO" else COLORS["gray"] for n in names]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Profit")
    ax.set_title("(a) Profit by Strategy")

    # 2. Efficiency (top right)
    ax = axes[0, 1]
    effs = [results[c]["market_metrics"]["efficiency"]["mean"] for c in configs]
    eff_stds = [results[c]["market_metrics"]["efficiency"]["std"] for c in configs]
    colors = [COLORS["ppo"] if "ppo" in c else COLORS["zic"] for c in configs]
    x = np.arange(len(labels))
    ax.bar(x, effs, yerr=eff_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("(b) Market Efficiency")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # 3. Volatility (bottom left)
    ax = axes[1, 0]
    vols = [results[c]["market_metrics"]["price_volatility"]["mean"] for c in configs]
    vol_stds = [results[c]["market_metrics"]["price_volatility"]["std"] for c in configs]
    colors = [COLORS["ppo"] if "ppo" in c else COLORS["zip"] for c in configs]
    ax.bar(x, vols, yerr=vol_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Volatility (%)")
    ax.set_title("(c) Price Volatility")

    # 4. Trades per period (bottom right)
    ax = axes[1, 1]
    trades = [results[c]["market_metrics"]["trades_per_period"]["mean"] for c in configs]
    trade_stds = [results[c]["market_metrics"]["trades_per_period"]["std"] for c in configs]
    colors = [COLORS["ppo"] if "ppo" in c else COLORS["Skeleton"] for c in configs]
    ax.bar(x, trades, yerr=trade_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Trades per Period")
    ax.set_title("(d) Trading Volume")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined metrics to {output_path}")


def main():
    """Generate all figures."""
    results = load_results()
    if results is None:
        return

    figures_dir = Path("paper/arxiv/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual figures
    generate_profit_bar_chart(results, figures_dir / "ppo_zi_profit_bar.pdf")
    generate_efficiency_comparison(results, figures_dir / "ppo_zi_efficiency.pdf")
    generate_volatility_comparison(results, figures_dir / "ppo_zi_volatility.pdf")

    # Generate combined figure
    generate_combined_metrics(results, figures_dir / "ppo_zi_combined.pdf")

    # Print summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    ppo_mix = results["ppo_mix"]
    zi = results["zi_only"]
    zic = results["zic_only"]
    zip_ = results["zip_only"]

    print("\nMarket-Level Metrics Comparison:")
    print(f"{'Market':<15} {'Efficiency':>12} {'Volatility':>12} {'V-Ineff':>10} {'Trades/P':>10}")
    print("-" * 60)
    for name, r in [("ZI only", zi), ("ZIC only", zic), ("ZIP only", zip_), ("PPO+mix", ppo_mix)]:
        m = r["market_metrics"]
        print(
            f"{name:<15} {m['efficiency']['mean']:>10.1f}% {m['price_volatility']['mean']:>10.1f}% "
            f"{m['v_inefficiency']['mean']:>10.2f} {m['trades_per_period']['mean']:>10.1f}"
        )

    print("\nProfit by Strategy (PPO+mix market):")
    for s, data in sorted(ppo_mix["profits"].items(), key=lambda x: -x[1]["mean"]):
        print(f"  {s}: {data['mean']:,.0f} +/- {data['std']:,.0f}")

    if "ppo_behavior" in ppo_mix:
        print("\nPPO Trading Behavior:")
        b = ppo_mix["ppo_behavior"]
        print(f"  Avg trade time: {b['avg_trade_time']:.1f} / 100 steps")
        print(f"  Early trades (t<=30): {b['early_trades_pct']:.1f}%")
        print(f"  Late trades (t>=70): {b['late_trades_pct']:.1f}%")

    print(f"\nFigures saved to: {figures_dir}/")


if __name__ == "__main__":
    main()

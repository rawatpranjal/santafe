#!/usr/bin/env python3
"""
Generate Part 2 Figures for Santa Fe Tournament Replication.

Creates 4 figures:
- Figure 2.1: Kaplan efficiency (mixed vs pure markets)
- Figure 2.2: Price autocorrelation by trader type
- Figure 2.3: Trading volume timing (closing panic)
- Figure 2.4: Trader hierarchy chart

Usage:
    python scripts/generate_part2_figures.py
    python scripts/generate_part2_figures.py --figure 2.1  # Generate specific figure
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from paper_style import COLORS, setup_style

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

# Apply shared style
setup_style()

# Environment codes
ENVIRONMENTS = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]
ENV_LOWER = [e.lower() for e in ENVIRONMENTS]


def load_csv_results(pattern: str) -> pd.DataFrame:
    """Load and concatenate CSV results matching pattern."""
    results_dir = Path("results")
    dfs = []
    for env in ENV_LOWER:
        path = results_dir / f"{pattern}_{env}" / "results.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["environment"] = env.upper()
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def run_self_play_period(strategy: str, seed: int, num_steps: int = 100) -> dict[str, Any]:
    """
    Run a single self-play period and extract per-step trade data.

    Returns dict with:
    - trade_times: list of timesteps when trades occurred
    - trade_prices: list of transaction prices
    - num_trades: total trades
    """
    num_buyers = 4
    num_sellers = 4
    num_tokens = 4
    price_min = 1
    price_max = 1000
    gametype = 6453

    # Token generation
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents
    agents = []
    buyer_valuations: dict[int, list[int]] = {}
    seller_costs: dict[int, list[int]] = {}

    # Buyers
    for i in range(num_buyers):
        player_id = i + 1
        tokens = token_gen.generate_tokens(is_buyer=True)
        buyer_valuations[player_id] = tokens
        agents.append(
            create_agent(
                strategy,
                player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=tokens,
                seed=seed + player_id,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=price_min,
                price_max=price_max,
            )
        )

    # Sellers
    for i in range(num_sellers):
        player_id = num_buyers + i + 1
        tokens = token_gen.generate_tokens(is_buyer=False)
        seller_costs[player_id] = tokens
        agents.append(
            create_agent(
                strategy,
                player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=tokens,
                seed=seed + player_id,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=price_min,
                price_max=price_max,
            )
        )

    # Initialize agents
    for agent in agents:
        agent.start_round(agent.valuations)
        agent.start_period(1)

    buyers = [a for a in agents if a.is_buyer]
    sellers = [a for a in agents if not a.is_buyer]

    # Create and run market
    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=num_steps,
        price_min=price_min,
        price_max=price_max,
        buyers=buyers,
        sellers=sellers,
        seed=seed,
    )

    while market.current_time < market.num_times:
        market.run_time_step()

    # Extract trade times and prices from orderbook
    trade_times = []
    trade_prices = []
    for t in range(1, num_steps + 1):
        price = int(market.orderbook.trade_price[t])
        if price > 0:
            trade_times.append(t)
            trade_prices.append(price)

    return {
        "trade_times": trade_times,
        "trade_prices": trade_prices,
        "num_trades": len(trade_times),
    }


def compute_autocorrelation(prices: list[int]) -> float:
    """Compute lag-1 autocorrelation of price sequence."""
    if len(prices) < 3:
        return np.nan
    prices_arr = np.array(prices, dtype=float)
    n = len(prices_arr)
    mean = np.mean(prices_arr)
    var = np.var(prices_arr)
    if var == 0:
        return 0.0
    autocorr = np.sum((prices_arr[:-1] - mean) * (prices_arr[1:] - mean)) / ((n - 1) * var)
    return float(autocorr)


# ============================================================================
# Figure 2.1: Kaplan Efficiency (Mixed vs Pure)
# ============================================================================


def generate_figure_2_1() -> None:
    """Generate Kaplan efficiency comparison: self-play vs mixed tournament."""
    print("Generating Figure 2.1: Kaplan Mixed vs Pure Efficiency...")

    # Load self-play Kaplan data
    self_play_df = load_csv_results("p2_self_kap")

    # Load round-robin mixed data (filter for Kaplan)
    rr_df = load_csv_results("p2_rr_mixed")

    if self_play_df.empty or rr_df.empty:
        print("  ERROR: Missing data files")
        return

    # Aggregate self-play efficiency by environment
    self_play_eff = self_play_df.groupby("environment")["efficiency"].mean().reindex(ENVIRONMENTS)

    # For round-robin, use market efficiency (same for all agents in a period)
    # Take first row per period since efficiency is market-level
    rr_market = rr_df.drop_duplicates(subset=["round", "period", "environment"])
    rr_eff = rr_market.groupby("environment")["efficiency"].mean().reindex(ENVIRONMENTS)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ENVIRONMENTS))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        self_play_eff.values,
        width,
        label="Self-Play (8 Kaplan)",
        color=COLORS["Kaplan"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        rr_eff.values,
        width,
        label="Round-Robin (Mixed)",
        color=COLORS["gray"],
        alpha=0.8,
    )

    ax.set_xlabel("Environment")
    ax.set_ylabel("Market Efficiency (%)")
    ax.set_title("Kaplan Strategy: Self-Play vs Mixed Tournament Efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(ENVIRONMENTS, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # Save
    output_path = Path("paper/arxiv/figures/kaplan_mixed_vs_pure.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Figure 2.2: Price Autocorrelation by Trader Type
# ============================================================================


def generate_figure_2_2(num_periods: int = 10) -> None:
    """Generate price autocorrelation comparison across trader types."""
    print(f"Generating Figure 2.2: Price Autocorrelation (running {num_periods} periods each)...")

    strategies = ["ZI", "ZIC", "ZIP", "Kaplan"]
    autocorrs: dict[str, list[float]] = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"  Running {strategy}...", end=" ", flush=True)
        for seed in range(1, num_periods + 1):
            result = run_self_play_period(strategy, seed * 100)
            if len(result["trade_prices"]) >= 3:
                ac = compute_autocorrelation(result["trade_prices"])
                if not np.isnan(ac):
                    autocorrs[strategy].append(ac)
        print(f"{len(autocorrs[strategy])} valid periods")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(strategies))
    means = [np.mean(autocorrs[s]) if autocorrs[s] else 0 for s in strategies]
    stds = [np.std(autocorrs[s]) if len(autocorrs[s]) > 1 else 0 for s in strategies]
    colors = [COLORS.get(s, COLORS["gray"]) for s in strategies]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")

    ax.set_xlabel("Trader Type")
    ax.set_ylabel("Lag-1 Price Autocorrelation")
    ax.set_title("Price Autocorrelation by Strategy (Self-Play)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.annotate(
            f"{mean:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean + std + 0.05),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save
    output_path = Path("paper/arxiv/figures/price_autocorrelation.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Figure 2.3: Trading Volume Timing (Closing Panic)
# ============================================================================


def generate_figure_2_3(num_periods: int = 10) -> None:
    """Generate trading volume timing comparison (early vs late trades)."""
    print(f"Generating Figure 2.3: Trading Volume Timing (running {num_periods} periods each)...")

    strategies = ["ZI", "ZIC", "ZIP", "Kaplan"]
    num_bins = 10
    timing_data: dict[str, np.ndarray] = {s: np.zeros(num_bins) for s in strategies}
    total_trades: dict[str, int] = {s: 0 for s in strategies}

    for strategy in strategies:
        print(f"  Running {strategy}...", end=" ", flush=True)
        for seed in range(1, num_periods + 1):
            result = run_self_play_period(strategy, seed * 100)
            for t in result["trade_times"]:
                bin_idx = min(int((t - 1) / 10), num_bins - 1)
                timing_data[strategy][bin_idx] += 1
                total_trades[strategy] += 1
        print(f"{total_trades[strategy]} trades")

    # Normalize to fractions
    for s in strategies:
        if total_trades[s] > 0:
            timing_data[s] = timing_data[s] / total_trades[s]

    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(num_bins)
    x_labels = [f"{i*10+1}-{(i+1)*10}" for i in range(num_bins)]

    for strategy in strategies:
        ax.plot(
            x,
            timing_data[strategy],
            marker="o",
            linewidth=2,
            markersize=6,
            label=strategy,
            color=COLORS.get(strategy, COLORS["gray"]),
        )

    ax.set_xlabel("Time Window (steps)")
    ax.set_ylabel("Fraction of Trades")
    ax.set_title("Trade Timing Distribution by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

    # Add reference line for uniform distribution
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="Uniform")

    plt.tight_layout()

    # Save
    output_path = Path("paper/arxiv/figures/trading_volume_timing.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Figure 2.4: Trader Hierarchy Chart
# ============================================================================


def generate_figure_2_4() -> None:
    """Generate trader hierarchy visualization from round-robin results."""
    print("Generating Figure 2.4: Trader Hierarchy Chart...")

    # Load round-robin results
    rr_df = load_csv_results("p2_rr_mixed")

    if rr_df.empty:
        print("  ERROR: Missing round-robin data")
        return

    # Calculate average profit by strategy
    strategy_profits = (
        rr_df.groupby("agent_type")["period_profit"].mean().sort_values(ascending=False)
    )

    # Create horizontal bar chart (pyramid style)
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = strategy_profits.index.tolist()
    profits = strategy_profits.values

    # Get colors
    colors = [COLORS.get(s, COLORS["gray"]) for s in strategies]

    y = np.arange(len(strategies))
    bars = ax.barh(y, profits, color=colors, alpha=0.8, edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Average Period Profit")
    ax.set_title("Trader Hierarchy: Average Profit in Round-Robin Tournament")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, profit in zip(bars, profits):
        width = bar.get_width()
        ax.annotate(
            f"{profit:.0f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5 if width >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
        )

    # Invert y-axis so best is at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    output_path = Path("paper/arxiv/figures/trader_hierarchy.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Part 2 Figures")
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        help="Which figure to generate: 2.1, 2.2, 2.3, 2.4, or all",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=10,
        help="Number of periods to run for figures 2.2 and 2.3",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Part 2 Figures")
    print("=" * 60)

    if args.figure in ("all", "2.1"):
        generate_figure_2_1()

    if args.figure in ("all", "2.2"):
        generate_figure_2_2(args.periods)

    if args.figure in ("all", "2.3"):
        generate_figure_2_3(args.periods)

    if args.figure in ("all", "2.4"):
        generate_figure_2_4()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

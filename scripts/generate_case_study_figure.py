#!/usr/bin/env python3
"""
Generate Mixed Market Case Study Figure.

Creates a multi-panel visualization showing all 8 strategies competing
in a single period, demonstrating their different behavioral patterns.

Usage:
    python scripts/generate_case_study_figure.py
    python scripts/generate_case_study_figure.py --seed 42
    python scripts/generate_case_study_figure.py --search-seeds
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, ".")

from paper_style import COLORS, setup_style

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_actual_surplus,
    calculate_allocative_efficiency,
    calculate_max_surplus,
    extract_trades_from_orderbook,
)
from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.visual_tracer import extract_market_timeline

# Apply shared style settings
setup_style()

# 8 strategies in tournament order
STRATEGIES = ["ZIC", "ZIP", "GD", "Kaplan", "Ringuette", "Skeleton", "Ledyard", "Markup"]


@dataclass
class TradeInfo:
    """Information about a trade."""

    time: int
    price: int
    buyer_type: str
    seller_type: str
    buyer_profit: int
    seller_profit: int


@dataclass
class PeriodData:
    """All data from a single period for visualization."""

    # Market state over time
    times: list[int]
    high_bids: list[int]
    low_asks: list[int]

    # Trades
    trades: list[TradeInfo]

    # Activity by strategy (time -> strategy -> count)
    bid_activity: dict[str, list[int]]
    ask_activity: dict[str, list[int]]

    # Cumulative profit by strategy over time
    cumulative_profits: dict[str, list[int]]

    # Final profits
    final_profits: dict[str, int]

    # Equilibrium info
    ce_price: int
    max_surplus: int
    efficiency: float


def run_period(seed: int, num_steps: int = 100) -> PeriodData:
    """
    Run a single period with 8 strategies and extract visualization data.

    Args:
        seed: Random seed
        num_steps: Number of time steps

    Returns:
        PeriodData with all visualization data
    """
    num_buyers = 8
    num_sellers = 8
    num_tokens = 4
    price_min = 1
    price_max = 1000
    gametype = 6453

    # Token generation
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents (1 buyer + 1 seller per strategy)
    buyer_types = STRATEGIES
    seller_types = STRATEGIES

    agents = []
    buyer_valuations: dict[int, list[int]] = {}
    seller_costs: dict[int, list[int]] = {}

    # Buyers
    for i, agent_type in enumerate(buyer_types):
        player_id = i + 1
        tokens = token_gen.generate_tokens(is_buyer=True)
        buyer_valuations[player_id] = tokens
        agents.append(
            create_agent(
                agent_type,
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
    for i, agent_type in enumerate(seller_types):
        player_id = num_buyers + i + 1
        tokens = token_gen.generate_tokens(is_buyer=False)
        seller_costs[i + 1] = tokens
        agents.append(
            create_agent(
                agent_type,
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

    # End period
    for agent in agents:
        agent.end_period()

    # Extract timeline using VisualTracer
    buyer_type_map = {i + 1: buyer_types[i] for i in range(num_buyers)}
    seller_type_map = {i + 1: seller_types[i] for i in range(num_sellers)}
    timeline = extract_market_timeline(market.orderbook, buyer_type_map, seller_type_map)

    # Extract visualization data
    times = []
    high_bids = []
    low_asks = []
    trades: list[TradeInfo] = []

    # Activity tracking
    bid_activity: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    ask_activity: dict[str, list[int]] = {s: [] for s in STRATEGIES}

    # Cumulative profit tracking
    cumulative_profits: dict[str, list[int]] = {s: [0] for s in STRATEGIES}
    current_profits: dict[str, int] = {s: 0 for s in STRATEGIES}

    # Track token indices for profit calculation
    buyer_token_idx: dict[int, int] = {i + 1: 0 for i in range(num_buyers)}
    seller_token_idx: dict[int, int] = {i + 1: 0 for i in range(num_sellers)}

    for record in timeline:
        times.append(record.time)
        high_bids.append(record.high_bid if record.high_bid > 0 else np.nan)
        low_asks.append(record.low_ask if record.low_ask > 0 else np.nan)

        # Count bid/ask activity per strategy
        step_bids: dict[str, int] = defaultdict(int)
        step_asks: dict[str, int] = defaultdict(int)

        for action in record.agent_actions:
            if action.action == "bid" and action.result == "winner":
                step_bids[action.agent_type] += 1
            elif action.action == "ask" and action.result == "winner":
                step_asks[action.agent_type] += 1

        for s in STRATEGIES:
            bid_activity[s].append(step_bids.get(s, 0))
            ask_activity[s].append(step_asks.get(s, 0))

        # Track trades
        if record.trade_occurred:
            # Find buyer and seller types from actions
            buyer_type = None
            seller_type = None
            buyer_id = None
            seller_id = None

            for action in record.agent_actions:
                if action.result == "TRADE":
                    if action.is_buyer:
                        buyer_type = action.agent_type
                        buyer_id = action.agent_id
                    else:
                        seller_type = action.agent_type
                        seller_id = action.agent_id

            if buyer_type and seller_type and buyer_id and seller_id:
                # Calculate profits
                b_idx = buyer_token_idx[buyer_id]
                s_idx = seller_token_idx[seller_id]

                buyer_val = buyer_valuations[buyer_id][b_idx]
                seller_cost = seller_costs[seller_id][s_idx]

                buyer_profit = buyer_val - record.trade_price
                seller_profit = record.trade_price - seller_cost

                current_profits[buyer_type] += buyer_profit
                current_profits[seller_type] += seller_profit

                buyer_token_idx[buyer_id] += 1
                seller_token_idx[seller_id] += 1

                trades.append(
                    TradeInfo(
                        time=record.time,
                        price=record.trade_price,
                        buyer_type=buyer_type,
                        seller_type=seller_type,
                        buyer_profit=buyer_profit,
                        seller_profit=seller_profit,
                    )
                )

        # Update cumulative profits
        for s in STRATEGIES:
            cumulative_profits[s].append(current_profits[s])

    # Calculate efficiency
    raw_trades = extract_trades_from_orderbook(market.orderbook, num_steps)
    buyer_vals_list = [buyer_valuations[i + 1] for i in range(num_buyers)]
    seller_costs_list = [seller_costs[i + 1] for i in range(num_sellers)]

    actual_surplus = calculate_actual_surplus(raw_trades, buyer_valuations, seller_costs)
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
    efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    # Calculate CE price (midpoint of supply/demand intersection)
    all_buyer_vals = sorted([v for vals in buyer_vals_list for v in vals], reverse=True)
    all_seller_costs = sorted([c for costs in seller_costs_list for c in costs])
    ce_price = (
        (all_buyer_vals[len(trades)] + all_seller_costs[len(trades)]) // 2
        if len(trades) > 0
        else 500
    )

    # Final profits
    final_profits = {s: 0 for s in STRATEGIES}
    for i, agent in enumerate(buyers):
        final_profits[buyer_types[i]] += agent.period_profit
    for i, agent in enumerate(sellers):
        final_profits[seller_types[i]] += agent.period_profit

    return PeriodData(
        times=times,
        high_bids=high_bids,
        low_asks=low_asks,
        trades=trades,
        bid_activity=bid_activity,
        ask_activity=ask_activity,
        cumulative_profits=cumulative_profits,
        final_profits=final_profits,
        ce_price=ce_price,
        max_surplus=max_surplus,
        efficiency=efficiency,
    )


def evaluate_seed(seed: int) -> dict[str, Any]:
    """Evaluate a seed for visualization quality."""
    data = run_period(seed)

    # Score based on:
    # 1. Number of trades (want 6-12)
    # 2. Number of different strategies trading
    # 3. Spread of trade times (not all early or all late)
    # 4. Presence of sniper trades (late trades by Kaplan/Ringuette)

    num_trades = len(data.trades)
    trading_strategies = set()
    late_trades = 0  # After step 70

    for trade in data.trades:
        trading_strategies.add(trade.buyer_type)
        trading_strategies.add(trade.seller_type)
        if trade.time > 70:
            late_trades += 1

    sniper_trades = sum(
        1
        for t in data.trades
        if t.buyer_type in ("Kaplan", "Ringuette") or t.seller_type in ("Kaplan", "Ringuette")
    )

    # Compute score
    trade_score = 10 if 6 <= num_trades <= 12 else max(0, 10 - abs(num_trades - 9))
    diversity_score = len(trading_strategies) * 2
    timing_score = min(late_trades, 3) * 3
    sniper_score = min(sniper_trades, 2) * 5

    total_score = trade_score + diversity_score + timing_score + sniper_score

    return {
        "seed": seed,
        "score": total_score,
        "num_trades": num_trades,
        "trading_strategies": len(trading_strategies),
        "late_trades": late_trades,
        "sniper_trades": sniper_trades,
        "efficiency": data.efficiency,
    }


def search_seeds(num_seeds: int = 100) -> list[dict[str, Any]]:
    """Search for good seeds for visualization."""
    results = []
    print(f"Searching {num_seeds} seeds...")

    for seed in range(1, num_seeds + 1):
        result = evaluate_seed(seed)
        results.append(result)
        if seed % 20 == 0:
            print(f"  {seed}/{num_seeds} done")

    # Sort by score
    results.sort(key=lambda x: -x["score"])
    return results


def generate_figure(data: PeriodData, output_path: Path, seed: int) -> None:
    """Generate the multi-panel case study figure."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 2], hspace=0.25, wspace=0.25)

    # Panel A: Price Tunnel (top left, larger)
    ax_tunnel = fig.add_subplot(gs[0, 0])

    # CE band
    ce_margin = 30
    ax_tunnel.axhspan(
        data.ce_price - ce_margin,
        data.ce_price + ce_margin,
        alpha=0.15,
        color="green",
        label=f"CE Band ({data.ce_price})",
    )
    ax_tunnel.axhline(data.ce_price, color="green", linestyle="--", linewidth=1, alpha=0.5)

    # Best bid/ask trajectories
    ax_tunnel.step(
        data.times, data.high_bids, where="post", color="blue", linewidth=1.5, label="Best Bid"
    )
    ax_tunnel.step(
        data.times, data.low_asks, where="post", color="red", linewidth=1.5, label="Best Ask"
    )

    # Trade markers colored by strategy
    for trade in data.trades:
        # Plot buyer side
        ax_tunnel.scatter(
            trade.time,
            trade.price,
            s=100,
            c=COLORS[trade.buyer_type],
            marker="^",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        # Plot seller side
        ax_tunnel.scatter(
            trade.time,
            trade.price,
            s=100,
            c=COLORS[trade.seller_type],
            marker="v",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

    ax_tunnel.set_xlabel("Time Step")
    ax_tunnel.set_ylabel("Price")
    ax_tunnel.set_title(f"(A) Price Convergence Tunnel (Efficiency: {data.efficiency:.1f}%)")
    ax_tunnel.set_xlim(0, 100)
    ax_tunnel.legend(loc="upper right")
    ax_tunnel.grid(True, alpha=0.3)

    # Panel B: Activity Heatmap (top right)
    ax_heat = fig.add_subplot(gs[0, 1])

    # Bin activity into 10-step windows
    num_bins = 10
    bin_size = 10
    activity_matrix = np.zeros((len(STRATEGIES), num_bins))

    for i, strategy in enumerate(STRATEGIES):
        for b in range(num_bins):
            start = b * bin_size
            end = (b + 1) * bin_size
            bid_count = sum(data.bid_activity[strategy][start:end])
            ask_count = sum(data.ask_activity[strategy][start:end])
            activity_matrix[i, b] = bid_count + ask_count

    im = ax_heat.imshow(activity_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax_heat.set_yticks(range(len(STRATEGIES)))
    ax_heat.set_yticklabels(STRATEGIES)
    ax_heat.set_xticks(range(num_bins))
    ax_heat.set_xticklabels([f"{i*10+1}-{(i+1)*10}" for i in range(num_bins)], rotation=45)
    ax_heat.set_xlabel("Time Window")
    ax_heat.set_title("(B) Strategy Activity (Winning Bids/Asks)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label("Count")

    # Panel C: Cumulative Profit (bottom, full width)
    ax_profit = fig.add_subplot(gs[1, :])

    for strategy in STRATEGIES:
        profits = data.cumulative_profits[strategy]
        # Extend to match times length + 1 (initial 0)
        ax_profit.plot(
            range(len(profits)),
            profits,
            color=COLORS[strategy],
            linewidth=1.5,
            label=f"{strategy} ({data.final_profits[strategy]:+})",
        )

    ax_profit.set_xlabel("Time Step")
    ax_profit.set_ylabel("Cumulative Profit")
    ax_profit.set_title("(C) Cumulative Profit by Strategy")
    ax_profit.set_xlim(0, 100)
    ax_profit.legend(loc="upper left", ncol=4)
    ax_profit.grid(True, alpha=0.3)
    ax_profit.axhline(0, color="black", linewidth=0.5)

    # Add trade markers on profit panel
    for trade in data.trades:
        ax_profit.axvline(trade.time, color="gray", alpha=0.3, linewidth=0.5)

    plt.suptitle(f"Mixed Market Case Study (Seed {seed}, {len(data.trades)} trades)")

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
    parser = argparse.ArgumentParser(description="Generate Mixed Market Case Study Figure")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the period (default: auto-search for good seed)",
    )
    parser.add_argument(
        "--search-seeds",
        action="store_true",
        help="Search for good seeds and print results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/case_study_mixed.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    if args.search_seeds:
        results = search_seeds(100)
        print("\nTop 10 seeds:")
        print(
            f"{'Seed':<6} {'Score':<6} {'Trades':<7} {'Strats':<7} {'Late':<5} {'Sniper':<7} {'Eff%':<6}"
        )
        print("-" * 50)
        for r in results[:10]:
            print(
                f"{r['seed']:<6} {r['score']:<6} {r['num_trades']:<7} "
                f"{r['trading_strategies']:<7} {r['late_trades']:<5} "
                f"{r['sniper_trades']:<7} {r['efficiency']:.1f}"
            )
        return

    # Select seed
    if args.seed is None:
        print("Searching for best seed...")
        results = search_seeds(50)
        best_seed = results[0]["seed"]
        print(f"Best seed: {best_seed} (score: {results[0]['score']})")
    else:
        best_seed = args.seed

    # Run period and generate figure
    print(f"\nGenerating figure with seed {best_seed}...")
    data = run_period(best_seed)

    print("\nPeriod summary:")
    print(f"  Trades: {len(data.trades)}")
    print(f"  Efficiency: {data.efficiency:.1f}%")
    print(f"  CE Price: {data.ce_price}")
    print("\nTrade details:")
    for i, trade in enumerate(data.trades, 1):
        print(
            f"  {i}. t={trade.time}: {trade.buyer_type}(B) <- ${trade.price} -> {trade.seller_type}(S)"
        )
    print("\nFinal profits:")
    for s in sorted(STRATEGIES, key=lambda x: -data.final_profits[x]):
        print(f"  {s}: {data.final_profits[s]:+}")

    generate_figure(data, args.output, best_seed)


if __name__ == "__main__":
    main()

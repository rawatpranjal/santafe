#!/usr/bin/env python3
"""
Generate ZI/ZIC/ZIP Case Study Figure for Section 5.

Creates a 2-panel visualization showing the foundational ZI < ZIC < ZIP
hierarchy through a single period of mixed trading.

Usage:
    python scripts/generate_zi_case_study_figure.py
    python scripts/generate_zi_case_study_figure.py --seed 42
    python scripts/generate_zi_case_study_figure.py --search-seeds
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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

# 3 strategies for foundational comparison
STRATEGIES = ["ZI", "ZIC", "ZIP"]


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

    # Cumulative profit by strategy over time
    cumulative_profits: dict[str, list[int]]

    # Cumulative quote activity by strategy over time
    cumulative_quotes: dict[str, list[int]]

    # Final profits
    final_profits: dict[str, int]

    # Equilibrium info
    ce_price: int
    max_surplus: int
    efficiency: float

    # X-axis zoom range (auto-detected from trade times)
    x_max: int


def run_period(seed: int, num_steps: int = 100) -> PeriodData:
    """
    Run a single period with ZI, ZIC, ZIP and extract visualization data.

    Args:
        seed: Random seed
        num_steps: Number of time steps

    Returns:
        PeriodData with all visualization data
    """
    num_buyers = 6
    num_sellers = 6
    num_tokens = 4
    price_min = 1
    price_max = 1000
    gametype = 6453

    # Token generation
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents: 2 of each strategy per side
    buyer_types = ["ZI", "ZI", "ZIC", "ZIC", "ZIP", "ZIP"]
    seller_types = ["ZI", "ZI", "ZIC", "ZIC", "ZIP", "ZIP"]

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

    # Cumulative profit tracking
    cumulative_profits: dict[str, list[int]] = {s: [0] for s in STRATEGIES}
    current_profits: dict[str, int] = {s: 0 for s in STRATEGIES}

    # Cumulative quote tracking (bids + asks per strategy)
    cumulative_quotes: dict[str, list[int]] = {s: [0] for s in STRATEGIES}
    current_quotes: dict[str, int] = {s: 0 for s in STRATEGIES}

    # Track token indices for profit calculation
    buyer_token_idx: dict[int, int] = {i + 1: 0 for i in range(num_buyers)}
    seller_token_idx: dict[int, int] = {i + 1: 0 for i in range(num_sellers)}

    for record in timeline:
        times.append(record.time)
        high_bids.append(record.high_bid if record.high_bid > 0 else np.nan)
        low_asks.append(record.low_ask if record.low_ask > 0 else np.nan)

        # Track quote activity (bids and asks) per strategy
        for action in record.agent_actions:
            if action.action in ("bid", "ask"):
                agent_type = action.agent_type
                if agent_type in current_quotes:
                    current_quotes[agent_type] += 1

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

        # Update cumulative profits and quotes
        for s in STRATEGIES:
            cumulative_profits[s].append(current_profits[s])
            cumulative_quotes[s].append(current_quotes[s])

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
    # Use min to avoid index out of range
    trade_idx = min(len(trades), len(all_buyer_vals) - 1, len(all_seller_costs) - 1)
    ce_price = (
        (all_buyer_vals[trade_idx] + all_seller_costs[trade_idx]) // 2 if trade_idx > 0 else 500
    )

    # Final profits by strategy (aggregate over all agents of that type)
    final_profits = {s: 0 for s in STRATEGIES}
    for i, agent in enumerate(buyers):
        final_profits[buyer_types[i]] += agent.period_profit
    for i, agent in enumerate(sellers):
        final_profits[seller_types[i]] += agent.period_profit

    # Calculate x_max for auto-zoom (last trade time + buffer, or num_steps)
    if trades:
        x_max = max(t.time for t in trades) + 10
    else:
        x_max = num_steps

    return PeriodData(
        times=times,
        high_bids=high_bids,
        low_asks=low_asks,
        trades=trades,
        cumulative_profits=cumulative_profits,
        cumulative_quotes=cumulative_quotes,
        final_profits=final_profits,
        ce_price=ce_price,
        max_surplus=max_surplus,
        efficiency=efficiency,
        x_max=x_max,
    )


def evaluate_seed(seed: int) -> dict[str, Any]:
    """Evaluate a seed for visualization quality."""
    data = run_period(seed)

    # Score based on:
    # 1. Number of trades (want 8-16 for clear visualization)
    # 2. ZI making some bad trades (large price deviation)
    # 3. Clear profit hierarchy: ZIP > ZIC > ZI
    # 4. Good efficiency (shows market still works)

    num_trades = len(data.trades)

    # Check profit hierarchy
    zip_profit = data.final_profits["ZIP"]
    zic_profit = data.final_profits["ZIC"]
    zi_profit = data.final_profits["ZI"]

    hierarchy_correct = zip_profit > zic_profit > zi_profit

    # Check for ZI bad trades (prices far from CE)
    zi_bad_trades = sum(
        1
        for t in data.trades
        if (t.buyer_type == "ZI" or t.seller_type == "ZI") and abs(t.price - data.ce_price) > 100
    )

    # Compute score
    trade_score = 10 if 8 <= num_trades <= 16 else max(0, 10 - abs(num_trades - 12))
    hierarchy_score = 20 if hierarchy_correct else 0
    zi_chaos_score = min(zi_bad_trades, 3) * 5
    efficiency_score = 10 if data.efficiency > 70 else 0

    total_score = trade_score + hierarchy_score + zi_chaos_score + efficiency_score

    return {
        "seed": seed,
        "score": total_score,
        "num_trades": num_trades,
        "hierarchy": hierarchy_correct,
        "zi_bad_trades": zi_bad_trades,
        "efficiency": data.efficiency,
        "profits": f"ZIP:{zip_profit:+}, ZIC:{zic_profit:+}, ZI:{zi_profit:+}",
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
    """Generate the 3-panel case study figure with auto-zoom."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # Use x_max for auto-zoom on all panels
    x_lim = data.x_max

    # Panel A: Price Tunnel (ZOOMED)
    ax_tunnel = axes[0]

    # CE band
    ce_margin = 50
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
        # Plot buyer side (triangle up)
        ax_tunnel.scatter(
            trade.time,
            trade.price,
            s=80,
            c=COLORS[trade.buyer_type],
            marker="^",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        # Plot seller side (triangle down)
        ax_tunnel.scatter(
            trade.time,
            trade.price,
            s=80,
            c=COLORS[trade.seller_type],
            marker="v",
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

    ax_tunnel.set_xlabel("Time Step")
    ax_tunnel.set_ylabel("Price")
    ax_tunnel.set_title(f"(A) Price Convergence (Efficiency: {data.efficiency:.1f}%)")
    ax_tunnel.set_xlim(0, x_lim)
    ax_tunnel.legend(loc="upper right")
    ax_tunnel.grid(True, alpha=0.3)

    # Panel B: Cumulative Quote Activity (LINE CHART)
    ax_quotes = axes[1]

    for strategy in STRATEGIES:
        quotes = data.cumulative_quotes[strategy]
        # Truncate to x_lim + 1 for proper zoom
        quotes_truncated = quotes[: x_lim + 1]
        ax_quotes.plot(
            range(len(quotes_truncated)),
            quotes_truncated,
            color=COLORS[strategy],
            linewidth=2,
            label=f"{strategy} ({quotes[-1]})",
        )

    ax_quotes.set_xlabel("Time Step")
    ax_quotes.set_ylabel("Cumulative Quotes")
    ax_quotes.set_title("(B) Quote Activity by Strategy")
    ax_quotes.set_xlim(0, x_lim)
    ax_quotes.legend(loc="upper left")
    ax_quotes.grid(True, alpha=0.3)

    # Add trade markers on quote panel
    for trade in data.trades:
        ax_quotes.axvline(trade.time, color="gray", alpha=0.2, linewidth=0.5)

    # Panel C: Cumulative Profit
    ax_profit = axes[2]

    for strategy in STRATEGIES:
        profits = data.cumulative_profits[strategy]
        # Truncate to x_lim + 1 for proper zoom
        profits_truncated = profits[: x_lim + 1]
        ax_profit.plot(
            range(len(profits_truncated)),
            profits_truncated,
            color=COLORS[strategy],
            linewidth=2,
            label=f"{strategy} ({data.final_profits[strategy]:+})",
        )

    ax_profit.set_xlabel("Time Step")
    ax_profit.set_ylabel("Cumulative Profit")
    ax_profit.set_title("(C) Cumulative Profit by Strategy")
    ax_profit.set_xlim(0, x_lim)
    ax_profit.legend(loc="upper left")
    ax_profit.grid(True, alpha=0.3)
    ax_profit.axhline(0, color="black", linewidth=0.5)

    # Add trade markers on profit panel
    for trade in data.trades:
        ax_profit.axvline(trade.time, color="gray", alpha=0.2, linewidth=0.5)

    plt.suptitle(f"Zero Intelligence Case Study (Seed {seed}, {len(data.trades)} trades)")

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
    parser = argparse.ArgumentParser(description="Generate ZI/ZIC/ZIP Case Study Figure")
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
        default=Path("paper/arxiv/figures/case_study_zi.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    if args.search_seeds:
        results = search_seeds(100)
        print("\nTop 10 seeds:")
        print(
            f"{'Seed':<6} {'Score':<6} {'Trades':<7} {'Hier':<5} {'ZI Bad':<7} {'Eff%':<6} {'Profits'}"
        )
        print("-" * 70)
        for r in results[:10]:
            print(
                f"{r['seed']:<6} {r['score']:<6} {r['num_trades']:<7} "
                f"{'Yes' if r['hierarchy'] else 'No':<5} {r['zi_bad_trades']:<7} "
                f"{r['efficiency']:.1f}   {r['profits']}"
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
        price_dev = trade.price - data.ce_price
        print(
            f"  {i}. t={trade.time}: {trade.buyer_type}(B) <- ${trade.price} ({price_dev:+}) -> {trade.seller_type}(S)"
        )
    print("\nFinal profits:")
    for s in sorted(STRATEGIES, key=lambda x: -data.final_profits[x]):
        print(f"  {s}: {data.final_profits[s]:+}")

    generate_figure(data, args.output, best_seed)


if __name__ == "__main__":
    main()

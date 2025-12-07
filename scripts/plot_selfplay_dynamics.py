#!/usr/bin/env python3
"""
Generate Selfplay Dynamics Figure for Section 5.

Creates a 1x5 grid showing ZI, ZIC1, ZIC2, ZIP1, ZIP2 in selfplay with:
- Price tunnel (all bids/asks + trade markers)
- Best bid/ask spread overlay

Usage:
    python scripts/plot_selfplay_dynamics.py
    python scripts/plot_selfplay_dynamics.py --seed 42
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

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

# Apply shared style settings
setup_style()

# 5 strategies for 1x5 grid
STRATEGIES = ["ZI", "ZIC1", "ZIC2", "ZIP1", "ZIP2"]


@dataclass
class QuoteRecord:
    """A single bid or ask quote."""

    time: int
    price: int
    is_bid: bool  # True for bid, False for ask


@dataclass
class TradeRecord:
    """A single trade."""

    time: int
    price: int


@dataclass
class PanelData:
    """Data for a single panel in the 1x5 grid."""

    strategy: str
    quotes: list[QuoteRecord] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    best_bids: list[tuple[int, int]] = field(default_factory=list)  # (time, price)
    best_asks: list[tuple[int, int]] = field(default_factory=list)  # (time, price)
    ce_price: int = 500
    efficiency: float = 0.0
    x_max: int = 100


def run_selfplay_period(strategy: str, seed: int, num_steps: int = 100) -> PanelData:
    """
    Run a single period of selfplay with given strategy.

    Args:
        strategy: The strategy type (ZI, ZIC, ZI2, ZIP)
        seed: Random seed
        num_steps: Number of time steps

    Returns:
        PanelData with all visualization data
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
        seller_costs[i + 1] = tokens
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

    # End period
    for agent in agents:
        agent.end_period()

    # Extract data from orderbook (using numpy arrays directly)
    panel = PanelData(strategy=strategy)
    ob = market.orderbook

    # Extract all quotes (bids and asks) from numpy arrays
    for t in range(1, num_steps + 1):
        # All bids at time t
        for buyer_id in range(1, num_buyers + 1):
            bid = int(ob.bids[buyer_id, t])
            if bid > 0:
                panel.quotes.append(QuoteRecord(time=t, price=bid, is_bid=True))
        # All asks at time t
        for seller_id in range(1, num_sellers + 1):
            ask = int(ob.asks[seller_id, t])
            if ask > 0:
                panel.quotes.append(QuoteRecord(time=t, price=ask, is_bid=False))

    # Extract trades
    for t in range(1, num_steps + 1):
        trade_price = int(ob.trade_price[t])
        if trade_price > 0:
            panel.trades.append(TradeRecord(time=t, price=trade_price))

    # Build best bid/ask trajectories from orderbook arrays
    for t in range(1, num_steps + 1):
        high_bid = int(ob.high_bid[t])
        low_ask = int(ob.low_ask[t])
        if high_bid > 0:
            panel.best_bids.append((t, high_bid))
        if low_ask > 0:
            panel.best_asks.append((t, low_ask))

    # Calculate efficiency
    raw_trades = extract_trades_from_orderbook(market.orderbook, num_steps)
    buyer_vals_list = [buyer_valuations[i + 1] for i in range(num_buyers)]
    seller_costs_list = [seller_costs[i + 1] for i in range(num_sellers)]

    actual_surplus = calculate_actual_surplus(raw_trades, buyer_valuations, seller_costs)
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
    panel.efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    # Calculate CE price
    all_buyer_vals = sorted([v for vals in buyer_vals_list for v in vals], reverse=True)
    all_seller_costs = sorted([c for costs in seller_costs_list for c in costs])
    num_trades = len(panel.trades)
    trade_idx = min(num_trades, len(all_buyer_vals) - 1, len(all_seller_costs) - 1)
    if trade_idx > 0:
        panel.ce_price = (all_buyer_vals[trade_idx] + all_seller_costs[trade_idx]) // 2
    else:
        panel.ce_price = 500

    # Auto-zoom x-axis
    if panel.trades:
        panel.x_max = max(t.time for t in panel.trades) + 10
    else:
        panel.x_max = num_steps

    return panel


def generate_figure(panels: list[PanelData], output_path: Path, seed: int) -> None:
    """Generate the 1x5 selfplay dynamics figure with enhanced aesthetics."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    strategy_labels = {
        "ZI": "ZI (Unconstrained)",
        "ZIC1": "ZIC1 (Budget)",
        "ZIC2": "ZIC2 (Market-Aware)",
        "ZIP1": "ZIP1 (Learning)",
        "ZIP2": "ZIP2 (Learning+Aware)",
    }

    # Panel background colors (subtle tints)
    bg_colors = {
        "ZI": "#ffebee",  # Light red (inefficient)
        "ZIC1": "#e3f2fd",  # Light blue
        "ZIC2": "#fff3e0",  # Light orange
        "ZIP1": "#e8f5e9",  # Light green
        "ZIP2": "#e8f5e9",  # Light green
    }

    for idx, (panel, ax) in enumerate(zip(panels, axes)):
        color = COLORS[panel.strategy]

        # Subtle background tint
        ax.set_facecolor(bg_colors.get(panel.strategy, "white"))

        # Plot all quotes as scatter (slightly larger, more visible)
        bid_times = [q.time for q in panel.quotes if q.is_bid]
        bid_prices = [q.price for q in panel.quotes if q.is_bid]
        ask_times = [q.time for q in panel.quotes if not q.is_bid]
        ask_prices = [q.price for q in panel.quotes if not q.is_bid]

        ax.scatter(
            bid_times,
            bid_prices,
            s=25,
            c="#0066FF",  # Vibrant blue for bids
            alpha=0.5,
            marker="^",
            label="Bids",
        )
        ax.scatter(
            ask_times,
            ask_prices,
            s=25,
            c="#FF0000",  # Vibrant red for asks
            alpha=0.5,
            marker="v",
            label="Asks",
        )

        # Plot best bid/ask lines (spread overlay) - thicker and vibrant
        if panel.best_bids:
            bid_t, bid_p = zip(*panel.best_bids)
            ax.step(bid_t, bid_p, where="post", color="#0066FF", linewidth=2.5, label="Best Bid")
        if panel.best_asks:
            ask_t, ask_p = zip(*panel.best_asks)
            ax.step(ask_t, ask_p, where="post", color="#FF0000", linewidth=2.5, label="Best Ask")

        # Plot trades as large markers (bigger, more prominent)
        trade_times = [t.time for t in panel.trades]
        trade_prices = [t.price for t in panel.trades]
        ax.scatter(
            trade_times,
            trade_prices,
            s=140,
            c=color,
            marker="o",
            edgecolors="white",
            linewidth=2,
            zorder=10,
            label=f"Trades ({len(panel.trades)})",
        )

        # CE line
        ax.axhline(
            panel.ce_price,
            color="#388E3C",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"CE ({panel.ce_price})",
        )

        # CE band (slightly more visible)
        ax.axhspan(
            panel.ce_price - 50,
            panel.ce_price + 50,
            alpha=0.12,
            color="#388E3C",
        )

        # Labels and styling
        label = strategy_labels.get(panel.strategy, panel.strategy)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Step", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Price", fontsize=10)
        ax.set_xlim(0, panel.x_max)
        ax.set_ylim(0, 1000)

        # Add efficiency as prominent text annotation
        text_color = "#D32F2F" if panel.efficiency < 50 else "#388E3C"
        ax.text(
            0.97,
            0.03,
            f"Eff: {panel.efficiency:.0f}%\nTrades: {len(panel.trades)}",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor=text_color, linewidth=1.5
            ),
        )

    plt.suptitle("Selfplay Market Dynamics", fontsize=13, fontweight="bold", y=1.02)
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
    parser = argparse.ArgumentParser(description="Generate Selfplay Dynamics Figure")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the period",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/selfplay_dynamics.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    print(f"Generating selfplay dynamics figure with seed {args.seed}...")

    panels = []
    for strategy in STRATEGIES:
        print(f"  Running {strategy} selfplay...")
        panel = run_selfplay_period(strategy, args.seed)
        panels.append(panel)
        print(f"    Efficiency: {panel.efficiency:.1f}%, Trades: {len(panel.trades)}")

    generate_figure(panels, args.output, args.seed)


if __name__ == "__main__":
    main()

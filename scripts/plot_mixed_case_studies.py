#!/usr/bin/env python3
"""
Generate Mixed Case Studies Figure for Section 5.

Creates a multi-panel figure showing ZI, ZIC, ZI2, ZIP in mixed competition
across multiple market environments to highlight different dynamics.

Environments:
- BASE: Standard environment
- SHRT: Short trading periods (20 steps, time pressure)
- RAN: Random token values
- SML: Small market (2 buyers, 2 sellers)
- BBBS: Many buyers, few sellers
- BSSS: Few buyers, many sellers

Usage:
    python scripts/plot_mixed_case_studies.py
    python scripts/plot_mixed_case_studies.py --seed 42
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

# 4 strategies in mixed competition
STRATEGIES = ["ZI", "ZIC", "ZI2", "ZIP"]

# Environment configurations
ENVIRONMENTS = {
    "BASE": {
        "num_buyers": 8,
        "num_sellers": 8,
        "num_tokens": 4,
        "num_steps": 100,
        "gametype": 6453,
        "label": "BASE (Standard)",
    },
    "SHRT": {
        "num_buyers": 8,
        "num_sellers": 8,
        "num_tokens": 4,
        "num_steps": 20,
        "gametype": 6453,
        "label": "SHRT (20 steps)",
    },
    "RAN": {
        "num_buyers": 8,
        "num_sellers": 8,
        "num_tokens": 4,
        "num_steps": 100,
        "gametype": 9999,  # Random tokens
        "label": "RAN (Random)",
    },
    "SML": {
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_steps": 100,
        "gametype": 6453,
        "label": "SML (2v2)",
    },
    "BBBS": {
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_steps": 100,
        "gametype": 6453,
        "label": "BBBS (6B 2S)",
    },
    "BSSS": {
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "num_steps": 100,
        "gametype": 6453,
        "label": "BSSS (2B 6S)",
    },
}


@dataclass
class TradeRecord:
    """A single trade."""

    time: int
    price: int
    buyer_type: str
    seller_type: str


@dataclass
class PanelData:
    """Data for a single panel (one environment)."""

    env_name: str
    env_label: str
    trades: list[TradeRecord] = field(default_factory=list)
    best_bids: list[tuple[int, int]] = field(default_factory=list)
    best_asks: list[tuple[int, int]] = field(default_factory=list)
    ce_price: int = 500
    efficiency: float = 0.0
    final_profits: dict[str, int] = field(default_factory=dict)
    x_max: int = 100


def run_mixed_period(env_name: str, seed: int) -> PanelData:
    """
    Run a single period of mixed competition in a given environment.

    Args:
        env_name: Name of the environment (BASE, SHRT, etc.)
        seed: Random seed

    Returns:
        PanelData with all visualization data
    """
    env = ENVIRONMENTS[env_name]
    num_buyers = env["num_buyers"]
    num_sellers = env["num_sellers"]
    num_tokens = env["num_tokens"]
    num_steps = env["num_steps"]
    gametype = env["gametype"]
    price_min = 1
    price_max = 1000

    # Token generation
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents: distribute strategies evenly among buyers and sellers
    # For each side, assign strategies in round-robin fashion
    agents = []
    buyer_valuations: dict[int, list[int]] = {}
    seller_costs: dict[int, list[int]] = {}
    buyer_types: dict[int, str] = {}
    seller_types: dict[int, str] = {}

    # Buyers
    for i in range(num_buyers):
        player_id = i + 1
        strategy = STRATEGIES[i % len(STRATEGIES)]
        buyer_types[player_id] = strategy
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

    # Sellers (seller_id is 1-indexed, player_id is num_buyers + seller_id)
    for i in range(num_sellers):
        seller_id = i + 1  # 1-indexed seller ID for efficiency.py
        player_id = num_buyers + seller_id  # actual player ID
        strategy = STRATEGIES[i % len(STRATEGIES)]
        seller_types[player_id] = strategy
        tokens = token_gen.generate_tokens(is_buyer=False)
        seller_costs[seller_id] = tokens  # Use seller_id (1..num_sellers) not player_id
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

    # Extract data
    panel = PanelData(env_name=env_name, env_label=env["label"])
    ob = market.orderbook

    # Extract trades with buyer/seller types
    for t in range(1, num_steps + 1):
        trade_price = int(ob.trade_price[t])
        if trade_price > 0:
            # Find buyer and seller who traded
            buyer_id = int(ob.high_bidder[t])
            seller_id = int(ob.low_asker[t])
            panel.trades.append(
                TradeRecord(
                    time=t,
                    price=trade_price,
                    buyer_type=buyer_types.get(buyer_id, "?"),
                    seller_type=seller_types.get(seller_id, "?"),
                )
            )

    # Build best bid/ask trajectories
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
    seller_costs_list = [
        seller_costs[i + 1] for i in range(num_sellers)
    ]  # seller_costs uses seller_id (1..num_sellers)

    actual_surplus = calculate_actual_surplus(raw_trades, buyer_valuations, seller_costs)
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
    panel.efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    # Calculate CE price
    all_buyer_vals = sorted([v for vals in buyer_vals_list for v in vals], reverse=True)
    all_seller_costs = sorted([c for costs in seller_costs_list for c in costs])
    num_trades_ce = len(panel.trades)
    trade_idx = min(num_trades_ce, len(all_buyer_vals) - 1, len(all_seller_costs) - 1)
    if trade_idx > 0:
        panel.ce_price = (all_buyer_vals[trade_idx] + all_seller_costs[trade_idx]) // 2
    else:
        panel.ce_price = 500

    # Aggregate profits by strategy
    panel.final_profits = {s: 0 for s in STRATEGIES}
    for agent in buyers:
        strategy = buyer_types[agent.player_id]
        panel.final_profits[strategy] += agent.period_profit
    for agent in sellers:
        strategy = seller_types[agent.player_id]
        panel.final_profits[strategy] += agent.period_profit

    # X-axis max
    panel.x_max = num_steps
    if panel.trades:
        panel.x_max = max(t.time for t in panel.trades) + 5

    return panel


def generate_figure(panels: list[PanelData], output_path: Path, seed: int) -> None:
    """Generate the multi-environment mixed case studies figure with enhanced aesthetics."""
    # 2x3 grid for 6 environments
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Environment background colors (subtle tints to distinguish panels)
    env_bg_colors = {
        "BASE": "#f5f5f5",  # Light gray
        "SHRT": "#ffebee",  # Light red (stress)
        "RAN": "#fff3e0",  # Light orange (stress)
        "SML": "#fce4ec",  # Light pink (stress)
        "BBBS": "#e3f2fd",  # Light blue
        "BSSS": "#e8f5e9",  # Light green
    }

    for idx, (panel, ax) in enumerate(zip(panels, axes)):
        # Subtle background tint
        ax.set_facecolor(env_bg_colors.get(panel.env_name, "white"))

        # Plot best bid/ask trajectories (thicker lines)
        if panel.best_bids:
            bid_t, bid_p = zip(*panel.best_bids)
            ax.step(
                bid_t,
                bid_p,
                where="post",
                color="#1976D2",
                linewidth=2.0,
                alpha=0.8,
                label="Best Bid",
            )
        if panel.best_asks:
            ask_t, ask_p = zip(*panel.best_asks)
            ax.step(
                ask_t,
                ask_p,
                where="post",
                color="#D32F2F",
                linewidth=2.0,
                alpha=0.8,
                label="Best Ask",
            )

        # CE line and band (more prominent)
        ax.axhline(
            panel.ce_price,
            color="#388E3C",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"CE ({panel.ce_price})",
        )
        ax.axhspan(panel.ce_price - 50, panel.ce_price + 50, alpha=0.12, color="#388E3C")

        # Plot trades colored by buyer type (larger markers with white edge)
        for trade in panel.trades:
            color = COLORS.get(trade.buyer_type, "#7f8c8d")
            ax.scatter(
                trade.time,
                trade.price,
                s=120,
                c=color,
                marker="o",
                edgecolors="white",
                linewidth=1.5,
                zorder=10,
            )

        # Labels and styling
        ax.set_title(f"{panel.env_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price")
        ax.set_xlim(0, panel.x_max)
        ax.set_ylim(0, 1000)

        # Efficiency annotation (prominent box in bottom-right)
        ax.text(
            0.98,
            0.02,
            f"Eff: {panel.efficiency:.0f}%",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="#388E3C", linewidth=1.5
            ),
        )

        # Profit summary annotation (cleaner, top-left)
        # Find winner strategy
        winner = max(STRATEGIES, key=lambda s: panel.final_profits[s])
        winner_profit = panel.final_profits[winner]
        ax.text(
            0.02,
            0.98,
            f"Winner: {winner} (+{winner_profit})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            color=COLORS.get(winner, "#333"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
        )

    # Create a shared legend outside the panels
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[s],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=s,
        )
        for s in STRATEGIES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=11,
        title="Buyer Strategy",
        title_fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.suptitle("Mixed Competition Case Studies", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

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
    parser = argparse.ArgumentParser(description="Generate Mixed Case Studies Figure")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the period",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/mixed_case_studies.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    print(f"Generating mixed case studies figure with seed {args.seed}...")

    panels = []
    for env_name in ["BASE", "SHRT", "RAN", "SML", "BBBS", "BSSS"]:
        print(f"  Running {env_name}...")
        panel = run_mixed_period(env_name, args.seed)
        panels.append(panel)
        print(f"    Efficiency: {panel.efficiency:.1f}%, Trades: {len(panel.trades)}")
        print(f"    Profits: {panel.final_profits}")

    generate_figure(panels, args.output, args.seed)


if __name__ == "__main__":
    main()

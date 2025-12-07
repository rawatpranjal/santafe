#!/usr/bin/env python3
"""
Generate Santa Fe Mixed-Play Dynamics Figure for Section 6.

Creates a single large panel showing trading dynamics from the mixed round-robin
tournament with all 12 Santa Fe traders (6 buyers + 6 sellers).

Usage:
    python scripts/plot_santafe_mixed_dynamics.py
    python scripts/plot_santafe_mixed_dynamics.py --round 1
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from paper_style import COLORS, setup_style

# Apply shared style settings
setup_style()

# Santa Fe 1991 roster - 12 traders
BUYER_LABELS = {1: "ZIC", 2: "Skeleton", 3: "Kaplan", 4: "Ringuette", 5: "Gamer", 6: "Perry"}
SELLER_LABELS = {7: "Ledyard", 8: "BGAN", 9: "Staecker", 10: "Jacobson", 11: "Lin", 12: "Breton"}

# All trader labels by agent_id
TRADER_LABELS = {**BUYER_LABELS, **SELLER_LABELS}

LOGS_DIR = Path("logs/p2_curated")


@dataclass
class QuoteRecord:
    """A single bid or ask quote."""

    time: int
    price: int
    is_bid: bool
    agent_id: int
    strategy: str


@dataclass
class TradeRecord:
    """A single trade."""

    time: int
    price: int
    buyer_id: int
    seller_id: int
    buyer_type: str
    seller_type: str


@dataclass
class PanelData:
    """Data for the mixed-play panel."""

    quotes: list[QuoteRecord] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    best_bids: list[tuple[int, int]] = field(default_factory=list)
    best_asks: list[tuple[int, int]] = field(default_factory=list)
    ce_price: int = 500
    max_surplus: int = 0
    actual_surplus: int = 0
    efficiency: float = 0.0
    x_max: int = 100


def load_events(log_path: Path) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def extract_panel_data(events: list[dict], target_round: int = 1) -> PanelData:
    """Extract visualization data from events for a specific round."""
    panel = PanelData()

    # Filter to target round, period 1
    round_events = [e for e in events if e.get("round") == target_round and e.get("period") == 1]

    # Get period info
    for e in round_events:
        if e.get("event_type") == "period_start":
            panel.ce_price = e.get("equilibrium_price", 500)
            panel.max_surplus = e.get("max_surplus", 0)
            break

    # Track best bid/ask over time
    best_bid: int = 0
    best_ask: int = 2000

    # Group events by step
    steps: dict[int, list[dict]] = {}
    for e in round_events:
        step = e.get("step", 0)
        if step not in steps:
            steps[step] = []
        steps[step].append(e)

    total_surplus = 0

    for step_num in sorted(steps.keys()):
        step_events = steps[step_num]

        step_best_bid = 0
        step_best_ask = 2000

        for e in step_events:
            if e.get("event_type") == "bid_ask":
                price = e.get("price", 0)
                is_bid = e.get("is_buyer", False)
                agent_id = e.get("agent_id", 0)
                status = e.get("status", "")

                # Get strategy name from agent_id
                strategy = TRADER_LABELS.get(agent_id, "Unknown")

                # Record quote
                if status != "pass" and price > 0:
                    panel.quotes.append(
                        QuoteRecord(
                            time=step_num,
                            price=price,
                            is_bid=is_bid,
                            agent_id=agent_id,
                            strategy=strategy,
                        )
                    )

                # Track best bid/ask
                if is_bid and price > step_best_bid:
                    step_best_bid = price
                elif not is_bid and price < step_best_ask:
                    step_best_ask = price

            elif e.get("event_type") == "trade":
                price = e.get("price", 0)
                buyer_id = e.get("buyer_id", 0)
                seller_id = e.get("seller_id", 0)
                buyer_profit = e.get("buyer_profit", 0)
                seller_profit = e.get("seller_profit", 0)

                buyer_type = BUYER_LABELS.get(buyer_id, e.get("buyer_type", "?"))
                seller_type = SELLER_LABELS.get(seller_id, e.get("seller_type", "?"))

                panel.trades.append(
                    TradeRecord(
                        time=step_num,
                        price=price,
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        buyer_type=buyer_type,
                        seller_type=seller_type,
                    )
                )
                total_surplus += buyer_profit + seller_profit

                # Reset best bid/ask after trade
                step_best_bid = 0
                step_best_ask = 2000

        # Update running best bid/ask
        if step_best_bid > 0:
            best_bid = step_best_bid
            panel.best_bids.append((step_num, best_bid))
        if step_best_ask < 2000:
            best_ask = step_best_ask
            panel.best_asks.append((step_num, best_ask))

    # Calculate efficiency
    panel.actual_surplus = total_surplus
    if panel.max_surplus > 0:
        panel.efficiency = 100 * total_surplus / panel.max_surplus

    # Auto-zoom x-axis
    if panel.trades:
        panel.x_max = max(t.time for t in panel.trades) + 10
    else:
        panel.x_max = 100

    return panel


def generate_figure(panel: PanelData, output_path: Path, target_round: int) -> None:
    """Generate the mixed-play dynamics figure (single large panel)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Light gray background
    ax.set_facecolor("#f8f9fa")

    # Plot best bid/ask trajectories
    if panel.best_bids:
        bid_t, bid_p = zip(*panel.best_bids)
        ax.step(
            bid_t, bid_p, where="post", color="#0066FF", linewidth=2.5, alpha=0.8, label="Best Bid"
        )
    if panel.best_asks:
        ask_t, ask_p = zip(*panel.best_asks)
        ax.step(
            ask_t, ask_p, where="post", color="#FF0000", linewidth=2.5, alpha=0.8, label="Best Ask"
        )

    # CE line and band
    ax.axhline(
        panel.ce_price,
        color="#388E3C",
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label=f"CE ({panel.ce_price})",
    )
    ax.axhspan(panel.ce_price - 50, panel.ce_price + 50, alpha=0.12, color="#388E3C")

    # Plot all quotes as small scatter points (background)
    for quote in panel.quotes:
        color = COLORS.get(quote.strategy, "#999999")
        marker = "^" if quote.is_bid else "v"
        ax.scatter(quote.time, quote.price, s=30, c=color, marker=marker, alpha=0.4, zorder=5)

    # Plot trades as large markers colored by buyer type
    for trade in panel.trades:
        color = COLORS.get(trade.buyer_type, "#757575")
        ax.scatter(
            trade.time,
            trade.price,
            s=180,
            c=color,
            marker="o",
            edgecolors="white",
            linewidth=2,
            zorder=10,
        )
        # Add small label for seller
        ax.annotate(
            trade.seller_type[:3],
            (trade.time, trade.price),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=7,
            alpha=0.7,
            ha="left",
        )

    # Labels and styling
    ax.set_title(
        f"Santa Fe Mixed Tournament (Round {target_round}, Period 1)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.set_xlim(0, panel.x_max)

    # Auto-scale y-axis
    all_prices = [q.price for q in panel.quotes] + [t.price for t in panel.trades]
    if all_prices:
        y_min = max(0, min(all_prices) - 50)
        y_max = max(all_prices) + 50
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0, 1000)

    # Efficiency annotation
    text_color = "#D32F2F" if panel.efficiency < 80 else "#388E3C"
    ax.text(
        0.98,
        0.02,
        f"Efficiency: {panel.efficiency:.0f}%\nTrades: {len(panel.trades)}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=text_color, linewidth=2),
    )

    # Create legend for all 12 traders
    # First row: Buyers
    buyer_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS.get(name, "#999"),
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"{name} (B)",
        )
        for id, name in sorted(BUYER_LABELS.items())
    ]
    # Second row: Sellers
    seller_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=COLORS.get(name, "#999"),
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"{name} (S)",
        )
        for id, name in sorted(SELLER_LABELS.items())
    ]

    # Combined legend
    all_handles = buyer_handles + seller_handles
    ax.legend(
        handles=all_handles,
        loc="upper left",
        ncol=3,
        fontsize=9,
        title="Santa Fe Traders",
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

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
    parser = argparse.ArgumentParser(description="Generate Santa Fe Mixed Dynamics Figure")
    parser.add_argument(
        "--round", type=int, default=1, help="Which round to visualize (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/santafe_mixed_dynamics.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    log_file = LOGS_DIR / "p2_rr_mixed_base_events.jsonl"

    if not log_file.exists():
        print(f"Error: Missing log file {log_file}")
        print("Run: uv run python scripts/generate_p2_event_logs.py mixed")
        return

    print(f"Generating Santa Fe mixed dynamics figure for round {args.round}...")
    print(f"  Loading from {log_file}...")

    events = load_events(log_file)
    panel = extract_panel_data(events, args.round)

    print(f"  Efficiency: {panel.efficiency:.1f}%")
    print(f"  Trades: {len(panel.trades)}")
    print(f"  CE Price: {panel.ce_price}")

    generate_figure(panel, args.output, args.round)


if __name__ == "__main__":
    main()

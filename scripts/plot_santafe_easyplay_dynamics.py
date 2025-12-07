#!/usr/bin/env python3
"""
Generate Santa Fe Easy-Play Dynamics Figure for Section 6.

Creates a 4x4 grid showing trading patterns from curated logs for all 13 Santa Fe traders
playing against TruthTeller sellers.

Usage:
    python scripts/plot_santafe_easyplay_dynamics.py
    python scripts/plot_santafe_easyplay_dynamics.py --round 1
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

# All 13 Santa Fe traders (12 original + ZIP) for 4x4 grid
# Organized by approximate ranking/importance
STRATEGIES = [
    ("Ringuette", "ring"),  # #1 rank
    ("Perry", "perry"),  # #2 rank
    ("Kaplan", "kap"),  # Original sniper
    ("Skeleton", "skel"),  # Evolutionary dominant
    ("BGAN", "bgan"),  # Santa Fe entrant
    ("Jacobson", "jacobson"),  # Santa Fe entrant
    ("Gamer", "gamer"),  # Santa Fe entrant
    ("Staecker", "staecker"),  # Santa Fe entrant
    ("Lin", "lin"),  # Santa Fe entrant
    ("Breton", "breton"),  # 99.8% efficiency
    ("Ledyard", "el"),  # Easley-Ledyard
    ("ZIC", "zic"),  # Zero-Intelligence Constrained
    ("ZIP", "zip"),  # Zero-Intelligence Plus
]

LOGS_DIR = Path("logs/p2_curated")


@dataclass
class QuoteRecord:
    """A single bid or ask quote."""

    time: int
    price: int
    is_bid: bool


@dataclass
class TradeRecord:
    """A single trade."""

    time: int
    price: int


@dataclass
class PanelData:
    """Data for a single panel in the grid."""

    strategy: str
    quotes: list[QuoteRecord] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    best_bids: list[tuple[int, int]] = field(default_factory=list)
    best_asks: list[tuple[int, int]] = field(default_factory=list)
    ce_price: int = 500
    max_surplus: int = 0
    actual_surplus: int = 0
    efficiency: float = 0.0
    x_max: int = 75


def load_events(log_path: Path) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def extract_panel_data(events: list[dict], strategy: str, target_round: int = 1) -> PanelData:
    """Extract visualization data from events for a specific round."""
    panel = PanelData(strategy=strategy)

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
                status = e.get("status", "")

                # Record quote
                if status != "pass" and price > 0:
                    panel.quotes.append(QuoteRecord(time=step_num, price=price, is_bid=is_bid))

                # Track best bid/ask
                if is_bid and price > step_best_bid:
                    step_best_bid = price
                elif not is_bid and price < step_best_ask:
                    step_best_ask = price

            elif e.get("event_type") == "trade":
                price = e.get("price", 0)
                buyer_profit = e.get("buyer_profit", 0)
                seller_profit = e.get("seller_profit", 0)

                panel.trades.append(TradeRecord(time=step_num, price=price))
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
        panel.x_max = 75

    return panel


def generate_figure(panels: list[PanelData], output_path: Path, target_round: int) -> None:
    """Generate the 4x4 easyplay dynamics figure (13 traders + 3 empty)."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    strategy_labels = {
        "Ringuette": "Ringuette (#1)",
        "Perry": "Perry (#2)",
        "Kaplan": "Kaplan (Sniper)",
        "Skeleton": "Skeleton",
        "BGAN": "BGAN",
        "Jacobson": "Jacobson",
        "Gamer": "Gamer",
        "Staecker": "Staecker",
        "Lin": "Lin",
        "Breton": "Breton",
        "Ledyard": "Easley-Ledyard",
        "ZIC": "ZIC",
        "ZIP": "ZIP",
    }

    # Panel background colors (light tints based on trader colors)
    bg_colors = {
        "Ringuette": "#fff3e0",  # Light orange
        "Perry": "#fce4ec",  # Light pink
        "Kaplan": "#ffebee",  # Light red
        "Skeleton": "#f3e5f5",  # Light purple
        "BGAN": "#e0f2f1",  # Light teal
        "Jacobson": "#eceff1",  # Light blue gray
        "Gamer": "#ede7f6",  # Light deep purple
        "Staecker": "#fbe9e7",  # Light deep orange
        "Lin": "#f1f8e9",  # Light light green
        "Breton": "#e8eaf6",  # Light indigo
        "Ledyard": "#e0f7fa",  # Light cyan
        "ZIC": "#e3f2fd",  # Light blue
        "ZIP": "#e8f5e9",  # Light green
    }

    for idx, (panel, ax) in enumerate(zip(panels, axes)):
        color = COLORS.get(panel.strategy, "#757575")

        # Subtle background tint
        ax.set_facecolor(bg_colors.get(panel.strategy, "white"))

        # Plot all quotes as scatter
        bid_times = [q.time for q in panel.quotes if q.is_bid]
        bid_prices = [q.price for q in panel.quotes if q.is_bid]
        ask_times = [q.time for q in panel.quotes if not q.is_bid]
        ask_prices = [q.price for q in panel.quotes if not q.is_bid]

        ax.scatter(bid_times, bid_prices, s=25, c="#0066FF", alpha=0.5, marker="^", label="Bids")
        ax.scatter(ask_times, ask_prices, s=25, c="#FF0000", alpha=0.5, marker="v", label="Asks")

        # Plot best bid/ask lines
        if panel.best_bids:
            bid_t, bid_p = zip(*panel.best_bids)
            ax.step(bid_t, bid_p, where="post", color="#0066FF", linewidth=2.5, label="Best Bid")
        if panel.best_asks:
            ask_t, ask_p = zip(*panel.best_asks)
            ax.step(ask_t, ask_p, where="post", color="#FF0000", linewidth=2.5, label="Best Ask")

        # Plot trades as large markers
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

        # CE line and band
        ax.axhline(
            panel.ce_price,
            color="#388E3C",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"CE ({panel.ce_price})",
        )
        ax.axhspan(panel.ce_price - 50, panel.ce_price + 50, alpha=0.12, color="#388E3C")

        # Labels and styling
        label = strategy_labels.get(panel.strategy, panel.strategy)
        ax.set_title(f"{label} vs TruthTeller", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time Step", fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel("Price", fontsize=9)
        ax.set_xlim(0, panel.x_max)

        # Auto-scale y-axis based on data
        all_prices = bid_prices + ask_prices + trade_prices
        if all_prices:
            y_min = max(0, min(all_prices) - 50)
            y_max = max(all_prices) + 50
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1000)

        # Add efficiency as text annotation
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

    # Hide empty axes (16 cells but only 13 traders)
    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"Santa Fe Easy-Play Dynamics (Round {target_round}, Period 1)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
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
    parser = argparse.ArgumentParser(description="Generate Santa Fe Easyplay Dynamics Figure")
    parser.add_argument(
        "--round", type=int, default=1, help="Which round to visualize (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/arxiv/figures/santafe_easyplay_dynamics.png"),
        help="Output path for figure",
    )

    args = parser.parse_args()

    print(f"Generating Santa Fe easyplay dynamics figure for round {args.round}...")

    panels = []
    for strategy_name, strategy_short in STRATEGIES:
        # Easy-play logs use p2_easy_* prefix
        log_file = LOGS_DIR / f"p2_easy_{strategy_short}_base_events.jsonl"

        if not log_file.exists():
            print(f"  Warning: Missing log file {log_file}")
            continue

        print(f"  Loading {strategy_name} from {log_file}...")
        events = load_events(log_file)
        panel = extract_panel_data(events, strategy_name, args.round)
        panels.append(panel)
        print(f"    Efficiency: {panel.efficiency:.1f}%, Trades: {len(panel.trades)}")

    if len(panels) < 13:
        print(f"\nWarning: Only {len(panels)} panels available, expected 13")

    generate_figure(panels, args.output, args.round)


if __name__ == "__main__":
    main()

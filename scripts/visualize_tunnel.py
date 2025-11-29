#!/usr/bin/env python3
"""
Convergence Tunnel Visualization.

Shows price discovery and spread convergence in a single market period.
The "tunnel" is formed by the best bid and best ask trajectories converging
toward the competitive equilibrium price.

Usage:
    python scripts/visualize_tunnel.py logs/exp_events.jsonl -r 1 -p 1
    python scripts/visualize_tunnel.py logs/exp_events.jsonl -o figures/tunnel.png
    python scripts/visualize_tunnel.py logs/old_events.jsonl --ce-price 150  # fallback
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from engine.event_logger import load_events


def plot_convergence_tunnel(
    events: list[dict],
    output_path: Path | None = None,
    round_filter: int | None = None,
    period_filter: int | None = None,
    ce_price_override: int | None = None,
    ce_margin: int = 10,
) -> None:
    """
    Plot Convergence Tunnel visualization.

    Args:
        events: List of event dictionaries from JSONL
        output_path: Path to save figure (if None, displays interactively)
        round_filter: Filter to specific round (if None, use first available)
        period_filter: Filter to specific period (if None, use first available)
        ce_price_override: Override equilibrium price (for old logs without period_start)
        ce_margin: Margin around CE price for band visualization (default: 10)
    """
    df = pd.DataFrame(events)

    if df.empty:
        print("No events found in log file.")
        return

    # Auto-select round/period if not specified
    if round_filter is None:
        round_filter = int(df["round"].min())
    if period_filter is None:
        period_filter = int(df["period"].min())

    # Get equilibrium price from period_start event or CLI override
    ce_price = ce_price_override
    if ce_price is None:
        period_starts = df[df["event_type"] == "period_start"]
        if not period_starts.empty:
            match = period_starts[
                (period_starts["round"] == round_filter)
                & (period_starts["period"] == period_filter)
            ]
            if not match.empty:
                ce_price = int(match.iloc[0]["equilibrium_price"])

    # Filter bid_ask and trade events
    bid_ask = df[df["event_type"] == "bid_ask"].copy()
    trades = df[df["event_type"] == "trade"].copy()

    # Apply round/period filters
    bid_ask = bid_ask[
        (bid_ask["round"] == round_filter) & (bid_ask["period"] == period_filter)
    ]
    trades = trades[
        (trades["round"] == round_filter) & (trades["period"] == period_filter)
    ]

    if bid_ask.empty:
        print(f"No bid/ask events for round={round_filter}, period={period_filter}")
        return

    # Separate buyers/sellers
    bids = bid_ask[bid_ask["is_buyer"] == True]
    asks = bid_ask[bid_ask["is_buyer"] == False]

    # Extract best bids/asks (status="winner" or "standing")
    # Winner = new best price this step, Standing = holding from previous step
    best_bids = bids[bids["status"].isin(["winner", "standing"])].copy()
    best_asks = asks[asks["status"].isin(["winner", "standing"])].copy()

    # Sort by step for line plots
    best_bids = best_bids.sort_values("step")
    best_asks = best_asks.sort_values("step")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # CE band (if available)
    if ce_price:
        ax.axhspan(
            ce_price - ce_margin,
            ce_price + ce_margin,
            alpha=0.2,
            color="green",
            label=f"CE Band ({ce_price}±{ce_margin})",
        )
        ax.axhline(ce_price, color="green", linestyle="--", linewidth=1.5, alpha=0.7)

    # All bids/asks (small dots)
    if not bids.empty:
        ax.scatter(
            bids["step"],
            bids["price"],
            s=15,
            alpha=0.3,
            c="blue",
            label=f"All bids (n={len(bids)})",
        )
    if not asks.empty:
        ax.scatter(
            asks["step"],
            asks["price"],
            s=15,
            alpha=0.3,
            c="red",
            label=f"All asks (n={len(asks)})",
        )

    # Best bid/ask trajectories (stepped lines)
    if not best_bids.empty:
        ax.step(
            best_bids["step"],
            best_bids["price"],
            where="post",
            color="blue",
            linewidth=2,
            label="Best bid",
        )
    if not best_asks.empty:
        ax.step(
            best_asks["step"],
            best_asks["price"],
            where="post",
            color="red",
            linewidth=2,
            label="Best ask",
        )

    # Trades (large markers)
    if not trades.empty:
        ax.scatter(
            trades["step"],
            trades["price"],
            s=150,
            c="green",
            marker="o",
            edgecolors="black",
            linewidth=1.5,
            label=f"Trades (n={len(trades)})",
            zorder=5,
        )

    # Labels and title
    title = f"Convergence Tunnel (Round {round_filter}, Period {period_filter})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits based on data
    all_prices = pd.concat([bids["price"], asks["price"]])
    if not all_prices.empty:
        y_min = max(0, all_prices.min() - 20)
        y_max = all_prices.max() + 20
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(events: list[dict], round_filter: int, period_filter: int) -> None:
    """Print summary of the selected period."""
    df = pd.DataFrame(events)

    # Get period_start info
    period_starts = df[df["event_type"] == "period_start"]
    if not period_starts.empty:
        match = period_starts[
            (period_starts["round"] == round_filter)
            & (period_starts["period"] == period_filter)
        ]
        if not match.empty:
            row = match.iloc[0]
            print(f"\n=== Period Info ===")
            print(f"Equilibrium Price: {row.get('equilibrium_price', 'N/A')}")
            print(f"Max Surplus: {row.get('max_surplus', 'N/A')}")

    # Trade summary
    trades = df[
        (df["event_type"] == "trade")
        & (df["round"] == round_filter)
        & (df["period"] == period_filter)
    ]
    if not trades.empty:
        print(f"\n=== Trade Summary ===")
        print(f"Trades: {len(trades)}")
        print(f"Avg price: {trades['price'].mean():.1f}")
        print(f"Price range: {trades['price'].min()} - {trades['price'].max()}")
        print(f"Price std: {trades['price'].std():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convergence Tunnel Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View with auto-detected CE price (new logs)
    python scripts/visualize_tunnel.py logs/exp_events.jsonl -r 1 -p 1

    # Save to file
    python scripts/visualize_tunnel.py logs/exp_events.jsonl -o figures/tunnel.png

    # Old logs without period_start event (manual CE price)
    python scripts/visualize_tunnel.py logs/old_events.jsonl --ce-price 150
        """,
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to JSONL event log file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--round",
        "-r",
        type=int,
        default=None,
        help="Filter to specific round (default: first available)",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=int,
        default=None,
        help="Filter to specific period (default: first available)",
    )
    parser.add_argument(
        "--ce-price",
        type=int,
        default=None,
        help="Override equilibrium price (for old logs without period_start)",
    )
    parser.add_argument(
        "--ce-margin",
        type=int,
        default=10,
        help="Margin around CE price for band visualization (default: 10)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics",
    )

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return

    # Load events
    print(f"Loading events from {args.log_file}...")
    events = load_events(args.log_file)
    print(f"Loaded {len(events)} events")

    # Auto-select round/period for summary
    df = pd.DataFrame(events)
    round_filter = args.round if args.round else int(df["round"].min())
    period_filter = args.period if args.period else int(df["period"].min())

    if args.summary:
        print_summary(events, round_filter, period_filter)

    # Plot
    plot_convergence_tunnel(
        events=events,
        output_path=args.output,
        round_filter=args.round,
        period_filter=args.period,
        ce_price_override=args.ce_price,
        ce_margin=args.ce_margin,
    )


if __name__ == "__main__":
    main()

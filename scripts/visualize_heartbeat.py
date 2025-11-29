#!/usr/bin/env python3
"""
Market Heartbeat Visualization.

Plots bidding activity frequency by trader type to reveal timing patterns:
- ZIC: uniform distribution (random firing)
- Kaplan: right-skewed (sniper behavior, waits then strikes)
- ZIP/GD: front-loaded (active early, tapers off)

Usage:
    python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl
    python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl --output figures/heartbeat.png
    python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl --period 1 --round 1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from engine.event_logger import load_events


def plot_market_heartbeat(
    events: list[dict[str, object]],
    output_path: Path | None = None,
    round_filter: int | None = None,
    period_filter: int | None = None,
    max_steps: int = 100,
) -> None:
    """
    Plot Market Heartbeat visualization.

    Args:
        events: List of event dictionaries from JSONL
        output_path: Path to save figure (if None, displays interactively)
        round_filter: Filter to specific round (if None, use all)
        period_filter: Filter to specific period (if None, use all)
        max_steps: Maximum time steps for x-axis
    """
    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Filter to bid/ask events only
    df = df[df["event_type"] == "bid_ask"]

    if df.empty:
        print("No bid/ask events found in log file.")
        return

    # Apply filters
    if round_filter is not None:
        df = df[df["round"] == round_filter]
    if period_filter is not None:
        df = df[df["period"] == period_filter]

    if df.empty:
        print(f"No events after filtering (round={round_filter}, period={period_filter})")
        return

    # Get unique trader types
    trader_types = sorted(df["agent_type"].unique())

    # Determine max_steps from data if not specified
    actual_max_steps = df["step"].max()
    if actual_max_steps > max_steps:
        max_steps = actual_max_steps

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette
    colors = plt.cm.tab10.colors

    # Plot histogram for each trader type
    bins = range(0, max_steps + 5, max(1, max_steps // 20))

    for i, trader_type in enumerate(trader_types):
        subset = df[df["agent_type"] == trader_type]
        color = colors[i % len(colors)]

        ax.hist(
            subset["step"],
            bins=bins,
            alpha=0.5,
            label=f"{trader_type} (n={len(subset)})",
            color=color,
            density=True,
        )

    # Add closing panic zone marker
    closing_zone = int(0.9 * max_steps)
    ax.axvline(
        x=closing_zone,
        linestyle="--",
        color="red",
        linewidth=2,
        label=f"Closing zone ({closing_zone})",
    )

    # Labels and title
    title = "Market Heartbeat: Bid/Ask Frequency by Trader Type"
    if round_filter or period_filter:
        filters = []
        if round_filter:
            filters.append(f"Round {round_filter}")
        if period_filter:
            filters.append(f"Period {period_filter}")
        title += f" ({', '.join(filters)})"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Bid/Ask Frequency (normalized)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(events: list[dict[str, object]]) -> None:
    """Print summary statistics of the event log."""
    df = pd.DataFrame(events)

    print("\n=== Event Log Summary ===")
    print(f"Total events: {len(df)}")

    if "event_type" in df.columns:
        print(f"\nEvent types:")
        for event_type, count in df["event_type"].value_counts().items():
            print(f"  {event_type}: {count}")

    bid_ask = df[df["event_type"] == "bid_ask"]
    if not bid_ask.empty:
        print(f"\nTrader types (bid/ask events):")
        for agent_type, count in bid_ask["agent_type"].value_counts().items():
            print(f"  {agent_type}: {count}")

        print(f"\nRounds: {bid_ask['round'].min()} - {bid_ask['round'].max()}")
        print(f"Periods: {bid_ask['period'].min()} - {bid_ask['period'].max()}")
        print(f"Steps: {bid_ask['step'].min()} - {bid_ask['step'].max()}")

    trades = df[df["event_type"] == "trade"]
    if not trades.empty:
        print(f"\nTrades: {len(trades)}")
        print(f"  Avg price: {trades['price'].mean():.2f}")
        print(f"  Price range: {trades['price'].min()} - {trades['price'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Market Heartbeat from event logs"
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to JSONL event log file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for figure (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--round", "-r",
        type=int,
        default=None,
        help="Filter to specific round",
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        default=None,
        help="Filter to specific period",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum time steps for x-axis (default: 100)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics only (no plot)",
    )

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return

    # Load events
    print(f"Loading events from {args.log_file}...")
    events = load_events(args.log_file)
    print(f"Loaded {len(events)} events")

    if args.summary:
        print_summary(events)
        return

    # Plot
    plot_market_heartbeat(
        events=events,
        output_path=args.output,
        round_filter=args.round,
        period_filter=args.period,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

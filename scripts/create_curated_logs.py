#!/usr/bin/env python3
"""Create human-readable curated trading logs from JSONL event logs.

Generates compact markdown files with one row per step, one column per agent.
Shows period 1 from 3 different rounds (different token values).

Outputs 3 consolidated files:
- easy_all.md: All easy-play strategies
- self_all.md: All self-play strategies
- mixed_all.md: Mixed all-vs-all

Usage:
    python scripts/create_curated_logs.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path("logs/p1_foundational")
OUTPUT_DIR = Path("logs/curated")

STRATEGIES = ["zi", "zic1", "zic2", "zip1", "zip2"]


def load_events(log_path: Path) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def filter_periods_from_different_rounds(events: list[dict], n: int = 3) -> list[dict]:
    """Filter to period 1 from first N rounds (different token values each)."""
    return [e for e in events if e.get("round", 0) <= n and e.get("period") == 1]


def format_trading_log(events: list[dict]) -> list[str]:
    """Format events as compact markdown with one column per agent.

    One row per step showing all agent bids/asks.
    """
    lines = []

    # Group by round
    rounds: dict[int, list[dict]] = defaultdict(list)
    round_info: dict[int, dict] = {}

    for e in events:
        if e.get("event_type") == "period_start":
            round_info[e.get("round", 1)] = e
        elif "round" in e:
            rounds[e["round"]].append(e)

    for round_num in sorted(rounds.keys()):
        info = round_info.get(round_num, {})
        eq_price = info.get("equilibrium_price", "?")
        max_surplus = info.get("max_surplus", "?")

        lines.append(f"### Round {round_num} (Period 1)")
        lines.append(f"**Equilibrium**: P*={eq_price}, Max Surplus={max_surplus}")
        lines.append("")

        # Group events by step
        steps: dict[int, list[dict]] = defaultdict(list)
        for e in rounds[round_num]:
            step = e.get("step", 0)
            steps[step].append(e)

        # Table header
        lines.append(
            "| Step | B1 | B2 | B3 | B4 | S5 | S6 | S7 | S8 | CurrBid | CurrAsk | Trade | Price | B.Prof | S.Prof |"
        )
        lines.append(
            "|------|-----|-----|-----|-----|-----|-----|-----|-----|---------|---------|-------|-------|--------|--------|"
        )

        total_surplus = 0
        prices = []
        trade_count = 0

        for step_num in sorted(steps.keys()):
            step_events = steps[step_num]

            # Collect bids/asks per agent
            agent_prices: dict[int, str] = {}
            winner_bid_agent = None
            winner_ask_agent = None
            curr_bid = "-"
            curr_ask = "-"

            for e in step_events:
                if e.get("event_type") == "bid_ask":
                    agent_id = e.get("agent_id", 0)
                    price = e.get("price", 0)
                    status = e.get("status", "")
                    is_buyer = e.get("is_buyer", True)

                    if status == "winner":
                        agent_prices[agent_id] = f"*{price}"
                        if is_buyer:
                            winner_bid_agent = agent_id
                            curr_bid = str(price)
                        else:
                            winner_ask_agent = agent_id
                            curr_ask = str(price)
                    elif status == "pass":
                        agent_prices[agent_id] = "-"
                    else:
                        agent_prices[agent_id] = str(price)

            # Get trade info
            trade_str = "-"
            price_str = "-"
            b_prof_str = "-"
            s_prof_str = "-"

            for e in step_events:
                if e.get("event_type") == "trade":
                    trade_count += 1
                    buyer_id = e.get("buyer_id", 0)
                    seller_id = e.get("seller_id", 0)
                    price = e.get("price", 0)
                    buyer_profit = e.get("buyer_profit", 0)
                    seller_profit = e.get("seller_profit", 0)

                    trade_str = f"B{buyer_id}→S{seller_id}"
                    price_str = str(price)
                    b_prof_str = str(buyer_profit)
                    s_prof_str = str(seller_profit)

                    if isinstance(buyer_profit, (int, float)) and isinstance(
                        seller_profit, (int, float)
                    ):
                        total_surplus += buyer_profit + seller_profit
                    if isinstance(price, (int, float)):
                        prices.append(price)

            # Build row
            b1 = agent_prices.get(1, "-")
            b2 = agent_prices.get(2, "-")
            b3 = agent_prices.get(3, "-")
            b4 = agent_prices.get(4, "-")
            s5 = agent_prices.get(5, "-")
            s6 = agent_prices.get(6, "-")
            s7 = agent_prices.get(7, "-")
            s8 = agent_prices.get(8, "-")

            lines.append(
                f"| {step_num} | {b1} | {b2} | {b3} | {b4} | {s5} | {s6} | {s7} | {s8} | {curr_bid} | {curr_ask} | {trade_str} | {price_str} | {b_prof_str} | {s_prof_str} |"
            )

        lines.append("")

        # Round summary
        efficiency = "?"
        volatility = "?"

        if max_surplus != "?" and max_surplus > 0:
            efficiency = f"{100 * total_surplus / max_surplus:.1f}%"

        if prices and eq_price != "?":
            mean_price = sum(prices) / len(prices)
            if mean_price > 0:
                std_price = math.sqrt(sum((p - mean_price) ** 2 for p in prices) / len(prices))
                volatility = f"{100 * std_price / mean_price:.1f}%"

        lines.append(
            f"**Summary**: {trade_count} trades | Efficiency: {efficiency} | Volatility: {volatility}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    return lines


def create_consolidated_easy_log() -> None:
    """Create single consolidated file with all easy-play strategies."""
    lines = [
        "# Easy-Play: All Strategies (BASE)",
        "",
        "**Setup**: Each strategy (4 buyers) vs TruthTeller (4 sellers), 4 tokens each",
        "",
    ]

    for strategy in STRATEGIES:
        log_path = LOGS_DIR / f"p1_easy_{strategy}_base_events.jsonl"
        if not log_path.exists():
            print(f"  Missing: {log_path}")
            continue

        events = load_events(log_path)
        events = filter_periods_from_different_rounds(events, 3)

        lines.append(f"## {strategy.upper()} vs TruthTeller")
        lines.append(f"4 {strategy.upper()} buyers vs 4 TruthTeller sellers")
        lines.append("")
        lines.extend(format_trading_log(events))

    output_path = OUTPUT_DIR / "easy_all.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def create_consolidated_self_log() -> None:
    """Create single consolidated file with all self-play strategies."""
    lines = [
        "# Self-Play: All Strategies (BASE)",
        "",
        "**Setup**: 8x same strategy (4 buyers + 4 sellers), 4 tokens each",
        "",
    ]

    for strategy in STRATEGIES:
        log_path = LOGS_DIR / f"p1_self_{strategy}_base_events.jsonl"
        if not log_path.exists():
            print(f"  Missing: {log_path}")
            continue

        events = load_events(log_path)
        events = filter_periods_from_different_rounds(events, 3)

        lines.append(f"## {strategy.upper()} (8x {strategy.upper()})")
        lines.append(f"4 {strategy.upper()} buyers vs 4 {strategy.upper()} sellers")
        lines.append("")
        lines.extend(format_trading_log(events))

    output_path = OUTPUT_DIR / "self_all.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def create_consolidated_mixed_log() -> None:
    """Create single consolidated file for mixed-play."""
    log_path = LOGS_DIR / "p1_mixed_base_events.jsonl"
    if not log_path.exists():
        print(f"  Missing: {log_path}")
        return

    events = load_events(log_path)
    events = filter_periods_from_different_rounds(events, 3)

    lines = [
        "# Mixed-Play: All-vs-All (BASE)",
        "",
        "**Setup**: Buyers: ZIC1(B1), ZIC2(B2), ZIP1(B3), ZIP2(B4) vs Sellers: ZIC1(S5), ZIC2(S6), ZIP1(S7), ZIP2(S8), 4 tokens each",
        "",
    ]
    lines.extend(format_trading_log(events))

    output_path = OUTPUT_DIR / "mixed_all.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def main():
    print("Creating consolidated curated trading logs...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Easy-Play (consolidated):")
    create_consolidated_easy_log()

    print("\nSelf-Play (consolidated):")
    create_consolidated_self_log()

    print("\nMixed-Play:")
    create_consolidated_mixed_log()

    print("\nDone! Created 3 consolidated curated logs.")


if __name__ == "__main__":
    main()

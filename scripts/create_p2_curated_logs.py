#!/usr/bin/env python3
"""Create human-readable curated trading logs from P2 Santa Fe JSONL event logs.

Generates compact markdown files with one row per step showing all 12 Santa Fe traders.
Shows period 1 from 3 different rounds (different token values).

Usage:
    python scripts/create_p2_curated_logs.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path("logs/p2_curated")
OUTPUT_DIR = Path("logs/curated")

# Santa Fe 1991 roster - 12 traders
BUYER_LABELS = {1: "ZIC", 2: "Skel", 3: "Kap", 4: "Ring", 5: "Gamer", 6: "Perry"}
SELLER_LABELS = {7: "Led", 8: "BGAN", 9: "Stae", 10: "Jacob", 11: "Lin", 12: "Bret"}


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


def format_trading_log_12traders(events: list[dict]) -> list[str]:
    """Format events as compact markdown with one column per agent (12 total).

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

        # Table header for 12 traders (6 buyers + 6 sellers)
        buyer_headers = " | ".join(BUYER_LABELS.values())
        seller_headers = " | ".join(SELLER_LABELS.values())
        lines.append(
            f"| Step | {buyer_headers} | {seller_headers} | Trade | Price | B.Prof | S.Prof |"
        )
        lines.append("|------|" + "------|" * 12 + "-------|-------|--------|--------|")

        total_surplus = 0
        prices = []
        trade_count = 0

        for step_num in sorted(steps.keys()):
            step_events = steps[step_num]

            # Collect bids/asks per agent
            agent_prices: dict[int, str] = {}

            for e in step_events:
                if e.get("event_type") == "bid_ask":
                    agent_id = e.get("agent_id", 0)
                    price = e.get("price", 0)
                    status = e.get("status", "")

                    if status == "winner":
                        agent_prices[agent_id] = f"**{price}**"
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
                    buyer_type = e.get("buyer_type", "?")
                    seller_type = e.get("seller_type", "?")
                    price = e.get("price", 0)
                    buyer_profit = e.get("buyer_profit", 0)
                    seller_profit = e.get("seller_profit", 0)

                    trade_str = f"{buyer_type[:3]}→{seller_type[:3]}"
                    price_str = str(price)
                    b_prof_str = str(buyer_profit)
                    s_prof_str = str(seller_profit)

                    if isinstance(buyer_profit, (int, float)) and isinstance(
                        seller_profit, (int, float)
                    ):
                        total_surplus += buyer_profit + seller_profit
                    if isinstance(price, (int, float)):
                        prices.append(price)

            # Build row with all 12 agents
            row_data = []
            for agent_id in range(1, 13):
                row_data.append(agent_prices.get(agent_id, "-"))

            buyer_cells = " | ".join(row_data[:6])
            seller_cells = " | ".join(row_data[6:])
            lines.append(
                f"| {step_num} | {buyer_cells} | {seller_cells} | {trade_str} | {price_str} | {b_prof_str} | {s_prof_str} |"
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


def create_p2_self_play_log() -> None:
    """Create curated logs for key Santa Fe self-play experiments."""
    lines = [
        "# P2 Santa Fe Tournament: Self-Play Logs (BASE)",
        "",
        "**Setup**: 8x same strategy (4 buyers + 4 sellers), 4 tokens each",
        "",
        "These logs show how each Santa Fe trader behaves in homogeneous self-play.",
        "",
    ]

    # All 12 Santa Fe traders + ZIP (Cliff 1997)
    for name, log_file in [
        # Original Santa Fe 1991 traders
        ("ZIC", "p2_self_zic_base_events.jsonl"),
        ("Skeleton", "p2_self_skel_base_events.jsonl"),
        ("Kaplan", "p2_self_kap_base_events.jsonl"),
        ("Ringuette", "p2_self_ring_base_events.jsonl"),
        ("Gamer", "p2_self_gamer_base_events.jsonl"),
        ("Perry", "p2_self_perry_base_events.jsonl"),
        ("Ledyard", "p2_self_el_base_events.jsonl"),
        ("BGAN", "p2_self_bgan_base_events.jsonl"),
        ("Staecker", "p2_self_staecker_base_events.jsonl"),
        ("Jacobson", "p2_self_jacobson_base_events.jsonl"),
        ("Lin", "p2_self_lin_base_events.jsonl"),
        ("Breton", "p2_self_breton_base_events.jsonl"),
        # Extended (post Santa Fe)
        ("ZIP", "p2_self_zip_base_events.jsonl"),
    ]:
        log_path = LOGS_DIR / log_file
        if not log_path.exists():
            print(f"  Missing: {log_path}")
            continue

        events = load_events(log_path)
        events = filter_periods_from_different_rounds(events, 3)

        lines.append(f"## {name} (Self-Play)")
        lines.append(f"4 {name} buyers vs 4 {name} sellers")
        lines.append("")

        # Use simpler 8-agent format for self-play
        lines.extend(format_self_play_log(events, name))

    output_path = OUTPUT_DIR / "s6_self.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def format_self_play_log(events: list[dict], strategy_name: str) -> list[str]:
    """Format self-play events (8 agents: 4 buyers + 4 sellers)."""
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
            "| Step | B1 | B2 | B3 | B4 | S5 | S6 | S7 | S8 | Trade | Price | B.Prof | S.Prof |"
        )
        lines.append(
            "|------|-----|-----|-----|-----|-----|-----|-----|-----|-------|-------|--------|--------|"
        )

        total_surplus = 0
        prices = []
        trade_count = 0

        for step_num in sorted(steps.keys()):
            step_events = steps[step_num]

            # Collect bids/asks per agent
            agent_prices: dict[int, str] = {}

            for e in step_events:
                if e.get("event_type") == "bid_ask":
                    agent_id = e.get("agent_id", 0)
                    price = e.get("price", 0)
                    status = e.get("status", "")

                    if status == "winner":
                        agent_prices[agent_id] = f"**{price}**"
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
            row_data = [agent_prices.get(i, "-") for i in range(1, 9)]
            lines.append(
                f"| {step_num} | {' | '.join(row_data)} | {trade_str} | {price_str} | {b_prof_str} | {s_prof_str} |"
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


def create_p2_mixed_log() -> None:
    """Create curated log for P2 mixed round-robin tournament."""
    log_path = LOGS_DIR / "p2_rr_mixed_base_events.jsonl"
    if not log_path.exists():
        print(f"  Missing: {log_path}")
        return

    events = load_events(log_path)
    events = filter_periods_from_different_rounds(events, 3)

    lines = [
        "# P2 Santa Fe Tournament: Mixed Round-Robin (BASE)",
        "",
        "**Setup**: All 12 Santa Fe 1991 traders (6 buyers + 6 sellers), 4 tokens each",
        "",
        "**Buyers**: ZIC(B1), Skeleton(B2), Kaplan(B3), Ringuette(B4), Gamer(B5), Perry(B6)",
        "",
        "**Sellers**: Ledyard(S7), BGAN(S8), Staecker(S9), Jacobson(S10), Lin(S11), Breton(S12)",
        "",
        "This is the canonical Santa Fe 1991 tournament configuration.",
        "",
    ]
    lines.extend(format_trading_log_12traders(events))

    output_path = OUTPUT_DIR / "s6_mixed.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def create_p2_easy_play_log() -> None:
    """Create curated logs for P2 Santa Fe easy-play experiments."""
    lines = [
        "# P2 Santa Fe Tournament: Easy-Play Logs (BASE)",
        "",
        "**Setup**: 4 buyers (strategy X) vs 4 TruthTeller sellers, 4 tokens each",
        "",
        "These logs show how each Santa Fe trader exploits passive sellers.",
        "",
    ]

    # All Santa Fe traders
    for name, log_file in [
        ("Ringuette", "p2_easy_ring_base_events.jsonl"),
        ("Perry", "p2_easy_perry_base_events.jsonl"),
        ("Kaplan", "p2_easy_kap_base_events.jsonl"),
        ("Skeleton", "p2_easy_skel_base_events.jsonl"),
        ("BGAN", "p2_easy_bgan_base_events.jsonl"),
        ("Jacobson", "p2_easy_jacobson_base_events.jsonl"),
        ("Gamer", "p2_easy_gamer_base_events.jsonl"),
        ("Staecker", "p2_easy_staecker_base_events.jsonl"),
        ("Lin", "p2_easy_lin_base_events.jsonl"),
        ("Breton", "p2_easy_breton_base_events.jsonl"),
        ("Ledyard", "p2_easy_el_base_events.jsonl"),
        ("ZIC", "p2_easy_zic_base_events.jsonl"),
        ("ZIP", "p2_easy_zip_base_events.jsonl"),
    ]:
        log_path = LOGS_DIR / log_file
        if not log_path.exists():
            print(f"  Missing: {log_path}")
            continue

        events = load_events(log_path)
        events = filter_periods_from_different_rounds(events, 3)

        lines.append(f"## {name} (Easy-Play)")
        lines.append(f"4 {name} buyers vs 4 TruthTeller sellers")
        lines.append("")

        # Use simpler 8-agent format for easy-play
        lines.extend(format_self_play_log(events, name))

    output_path = OUTPUT_DIR / "s6_easy.md"
    output_path.write_text("\n".join(lines))
    print(f"  Created: {output_path}")


def main():
    print("Creating P2 Santa Fe curated trading logs...")
    print(f"Input directory: {LOGS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Self-Play logs:")
    create_p2_self_play_log()

    print("\nEasy-Play logs:")
    create_p2_easy_play_log()

    print("\nMixed Round-Robin logs:")
    create_p2_mixed_log()

    print("\nDone!")


if __name__ == "__main__":
    main()

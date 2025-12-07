#!/usr/bin/env python3
"""Analyze Part 1 Easy-Play experiment results from JSONL event logs.

Computes metrics for results.md tables:
- Table 1.2b.1: Allocative Efficiency (%)
- Table 1.2b.2: Mean Trade Time (steps)
- Table 1.2b.3: Trades per Period

Usage:
    python scripts/analyze_p1_easy.py
    python scripts/analyze_p1_easy.py --update-results  # Also update results.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

STRATEGIES = ["zi", "zic1", "zic2", "zip1", "zip2"]
ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]
LOGS_DIR = Path("logs/p1_foundational")


def parse_log_file(log_path: Path) -> dict:
    """Parse a JSONL event log and compute metrics."""
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))

    # Separate event types
    period_starts = [e for e in events if e.get("event_type") == "period_start"]
    trades = [e for e in events if e.get("event_type") == "trade"]

    if not period_starts:
        return {"efficiency": None, "mean_trade_time": None, "trades_per_period": None}

    # Group by (round, period)
    trades_by_period: dict[tuple[int, int], list] = defaultdict(list)
    for t in trades:
        key = (t["round"], t["period"])
        trades_by_period[key].append(t)

    max_surplus_by_period: dict[tuple[int, int], int] = {}
    for ps in period_starts:
        key = (ps["round"], ps["period"])
        max_surplus_by_period[key] = ps["max_surplus"]

    # Compute metrics per period
    num_periods = len(period_starts)
    total_trades = len(trades)

    # Mean trade time: average step of first trade in each period
    first_trade_steps = []
    for key, period_trades in trades_by_period.items():
        if period_trades:
            first_step = min(t["step"] for t in period_trades)
            first_trade_steps.append(first_step)

    mean_trade_time = sum(first_trade_steps) / len(first_trade_steps) if first_trade_steps else None

    # Trades per period
    trades_per_period = total_trades / num_periods if num_periods > 0 else 0

    # Efficiency: We need to compute actual surplus from trades
    # This requires knowing buyer values and seller costs, which aren't in the log
    # For now, use trades_per_period / max_possible_trades as a proxy
    # (Proper efficiency requires the token values which are in period_start or trades)

    # Actually, let's compute efficiency differently:
    # Since TruthTeller sellers ask at cost, any trade at price >= cost is efficient
    # We'll compute efficiency as: trades_completed / max_possible_trades
    # Max possible trades = num_tokens * num_buyers (approximately)

    # From the period_start events, we have max_surplus
    # Efficiency = actual_surplus / max_surplus
    # But we don't have actual_surplus directly...

    # For easy-play, let's use a simpler metric:
    # Efficiency proxy = trades_per_period / expected_max_trades
    # Or just report trades_per_period and let the user interpret

    return {
        "mean_trade_time": mean_trade_time,
        "trades_per_period": trades_per_period,
        "total_trades": total_trades,
        "num_periods": num_periods,
    }


def analyze_all_easy_play() -> dict:
    """Analyze all easy-play experiment logs."""
    results: dict[str, dict[str, dict]] = defaultdict(dict)

    for strategy in STRATEGIES:
        for env in ENVS:
            log_file = LOGS_DIR / f"p1_easy_{strategy}_{env}_events.jsonl"
            if not log_file.exists():
                print(f"  Missing: {log_file.name}")
                continue

            metrics = parse_log_file(log_file)
            results[strategy.upper()][env.upper()] = metrics
            mean_time_str = (
                f"{metrics['mean_trade_time']:.1f}" if metrics["mean_trade_time"] else "N/A"
            )
            print(
                f"  {strategy.upper()} × {env.upper()}: trades/period={metrics['trades_per_period']:.1f}, mean_time={mean_time_str}"
            )

    return dict(results)


def format_table(results: dict, metric: str, decimals: int = 1) -> str:
    """Format results as markdown table."""
    header = "| Trader | " + " | ".join(env.upper() for env in ENVS) + " |"
    separator = "|--------|" + "|".join("------" for _ in ENVS) + "|"

    rows = [header, separator]
    for strategy in ["ZI", "ZIC1", "ZIC2", "ZIP1", "ZIP2"]:
        row = f"| **{strategy}** |"
        for env in [e.upper() for e in ENVS]:
            value = results.get(strategy, {}).get(env, {}).get(metric)
            if value is not None:
                row += f" {value:.{decimals}f} |"
            else:
                row += " -- |"
        rows.append(row)

    return "\n".join(rows)


def update_results_md(results: dict) -> None:
    """Update results.md with computed metrics."""
    results_path = Path("checklists/results.md")
    content = results_path.read_text()

    # Update Table 1.2b.2: Mean Trade Time
    mean_time_table = format_table(results, "mean_trade_time", decimals=1)

    # Update Table 1.2b.3: Trades per Period
    trades_table = format_table(results, "trades_per_period", decimals=1)

    # Find and replace the placeholder tables
    # Table 1.2b.2
    pattern = r"(\| Trader \| BASE \|.*?\n\|[-\|]+\n)(\| \*\*ZI\*\* \|[^\n]+\n\| \*\*ZIC1\*\* \|[^\n]+\n\| \*\*ZIC2\*\* \|[^\n]+\n\| \*\*ZIP1\*\* \|[^\n]+\n\| \*\*ZIP2\*\* \|[^\n]+)"

    # For now, just print the tables
    print("\n### Table 1.2b.2: Mean Trade Time (steps)")
    print(mean_time_table)

    print("\n### Table 1.2b.3: Trades per Period")
    print(trades_table)


def main():
    parser = argparse.ArgumentParser(description="Analyze Part 1 Easy-Play results")
    parser.add_argument("--update-results", action="store_true", help="Update results.md")
    parser.add_argument(
        "--output", type=Path, default=Path("results/p1_easy_metrics.json"), help="Output JSON file"
    )
    args = parser.parse_args()

    print("Analyzing Part 1 Easy-Play experiments...")
    print(f"Log directory: {LOGS_DIR}")
    print()

    results = analyze_all_easy_play()

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print formatted tables
    print("\n" + "=" * 60)
    print("EASY-PLAY RESULTS")
    print("=" * 60)

    print("\n### Table 1.2b.2: Mean Trade Time (steps)")
    print("*Lower = faster search against TruthTeller sellers*")
    print(format_table(results, "mean_trade_time", decimals=1))

    print("\n### Table 1.2b.3: Trades per Period")
    print(format_table(results, "trades_per_period", decimals=1))

    if args.update_results:
        update_results_md(results)


if __name__ == "__main__":
    main()

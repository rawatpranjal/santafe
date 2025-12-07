#!/usr/bin/env python3
"""Analyze Part 1 Self-Play experiment results from JSONL event logs.

Computes metrics for results.md tables:
- Table 1.3.7: Trades per Period

Usage:
    python scripts/analyze_p1_self.py
    python scripts/analyze_p1_self.py --update-results  # Also update results.md
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
        return {"trades_per_period": None, "total_trades": None, "num_periods": None}

    # Group by (round, period)
    trades_by_period: dict[tuple[int, int], list] = defaultdict(list)
    for t in trades:
        key = (t["round"], t["period"])
        trades_by_period[key].append(t)

    # Compute metrics per period
    num_periods = len(period_starts)
    total_trades = len(trades)

    # Trades per period
    trades_per_period = total_trades / num_periods if num_periods > 0 else 0

    return {
        "trades_per_period": trades_per_period,
        "total_trades": total_trades,
        "num_periods": num_periods,
    }


def analyze_all_self_play() -> dict:
    """Analyze all self-play experiment logs."""
    results: dict[str, dict[str, dict]] = defaultdict(dict)

    for strategy in STRATEGIES:
        for env in ENVS:
            log_file = LOGS_DIR / f"p1_self_{strategy}_{env}_events.jsonl"
            if not log_file.exists():
                print(f"  Missing: {log_file.name}")
                continue

            metrics = parse_log_file(log_file)
            results[strategy.upper()][env.upper()] = metrics
            tpp_str = (
                f"{metrics['trades_per_period']:.1f}" if metrics["trades_per_period"] else "N/A"
            )
            print(f"  {strategy.upper()} × {env.upper()}: trades/period={tpp_str}")

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


def main():
    parser = argparse.ArgumentParser(description="Analyze Part 1 Self-Play results")
    parser.add_argument("--update-results", action="store_true", help="Update results.md")
    parser.add_argument(
        "--output", type=Path, default=Path("results/p1_self_metrics.json"), help="Output JSON file"
    )
    args = parser.parse_args()

    print("Analyzing Part 1 Self-Play experiments...")
    print(f"Log directory: {LOGS_DIR}")
    print()

    results = analyze_all_self_play()

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print formatted tables
    print("\n" + "=" * 60)
    print("SELF-PLAY RESULTS")
    print("=" * 60)

    print("\n### Table 1.3.7: Trades per Period")
    print(format_table(results, "trades_per_period", decimals=1))


if __name__ == "__main__":
    main()

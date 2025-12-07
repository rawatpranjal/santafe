#!/usr/bin/env python3
"""Analyze Part 1 Mixed-Play experiment results from CSV files.

Computes metrics for results.md tables:
- Table 1.6.1: Average Profit by Strategy
- Table 1.6.2: Average Rank by Strategy
- Table 1.6.3: Win Rate (%) by Strategy
- Table 1.6.4: Profit per Trade by Strategy
- Analysis: Institutional Blindness Gap (ZIP2 - ZIP1)
- Analysis: Market Awareness Gap (ZIC2 - ZIC1)

Usage:
    python scripts/analyze_p1_mixed.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]
BUYER_STRATEGIES = ["ZIC1", "ZIC2", "ZIP1", "ZIP2"]
RESULTS_DIR = Path("results")


def analyze_mixed_experiment(csv_path: Path) -> dict:
    """Analyze a single mixed-play experiment from CSV."""
    df = pd.read_csv(csv_path)

    # Filter to buyers only (the 4 heterogeneous strategies)
    buyers = df[df["is_buyer"] == True].copy()

    # Group by round, period to compute ranks within each period
    results_by_strategy: dict[str, dict] = {
        s: {
            "profits": [],
            "ranks": [],
            "wins": 0,
            "total_periods": 0,
            "trades": [],
            "profit_per_trade": [],
        }
        for s in BUYER_STRATEGIES
    }

    # Process each period
    for (round_num, period_num), period_df in buyers.groupby(["round", "period"]):
        # Sort by profit descending to get ranks
        period_df = period_df.sort_values("period_profit", ascending=False)
        period_df["rank"] = range(1, len(period_df) + 1)

        for _, row in period_df.iterrows():
            strategy = row["agent_type"]
            if strategy in BUYER_STRATEGIES:
                results_by_strategy[strategy]["profits"].append(row["period_profit"])
                results_by_strategy[strategy]["ranks"].append(row["rank"])
                results_by_strategy[strategy]["total_periods"] += 1
                results_by_strategy[strategy]["trades"].append(row["num_trades"])
                if row["num_trades"] > 0:
                    results_by_strategy[strategy]["profit_per_trade"].append(
                        row["period_profit"] / row["num_trades"]
                    )
                if row["rank"] == 1:
                    results_by_strategy[strategy]["wins"] += 1

    # Compute aggregated metrics
    metrics = {}
    for strategy in BUYER_STRATEGIES:
        data = results_by_strategy[strategy]
        if data["profits"]:
            metrics[strategy] = {
                "avg_profit": np.mean(data["profits"]),
                "avg_rank": np.mean(data["ranks"]),
                "win_rate": (
                    100 * data["wins"] / data["total_periods"] if data["total_periods"] > 0 else 0
                ),
                "profit_per_trade": (
                    np.mean(data["profit_per_trade"]) if data["profit_per_trade"] else 0
                ),
            }

    return metrics


def analyze_all_mixed() -> dict:
    """Analyze all mixed-play experiments."""
    results: dict[str, dict[str, dict]] = {}

    for env in ENVS:
        csv_path = RESULTS_DIR / f"p1_mixed_{env}" / "results.csv"
        if not csv_path.exists():
            print(f"  Missing: {csv_path}")
            continue

        metrics = analyze_mixed_experiment(csv_path)
        results[env.upper()] = metrics

        # Print summary
        print(f"  {env.upper()}:", end="")
        for s in BUYER_STRATEGIES:
            if s in metrics:
                print(f" {s}={metrics[s]['avg_profit']:.0f}", end="")
        print()

    return results


def format_profit_table(results: dict) -> str:
    """Format average profit table."""
    header = "| Env | " + " | ".join(BUYER_STRATEGIES) + " |"
    separator = "|-----|" + "|".join("------" for _ in BUYER_STRATEGIES) + "|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        row = f"| {env} |"
        for strategy in BUYER_STRATEGIES:
            value = results.get(env, {}).get(strategy, {}).get("avg_profit")
            if value is not None:
                row += f" {value:.0f} |"
            else:
                row += " -- |"
        rows.append(row)

    return "\n".join(rows)


def format_rank_table(results: dict) -> str:
    """Format average rank table."""
    header = "| Env | " + " | ".join(BUYER_STRATEGIES) + " |"
    separator = "|-----|" + "|".join("------" for _ in BUYER_STRATEGIES) + "|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        row = f"| {env} |"
        for strategy in BUYER_STRATEGIES:
            value = results.get(env, {}).get(strategy, {}).get("avg_rank")
            if value is not None:
                row += f" {value:.1f} |"
            else:
                row += " -- |"
        rows.append(row)

    return "\n".join(rows)


def format_winrate_table(results: dict) -> str:
    """Format win rate table."""
    header = "| Env | " + " | ".join(BUYER_STRATEGIES) + " |"
    separator = "|-----|" + "|".join("------" for _ in BUYER_STRATEGIES) + "|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        row = f"| {env} |"
        for strategy in BUYER_STRATEGIES:
            value = results.get(env, {}).get(strategy, {}).get("win_rate")
            if value is not None:
                row += f" {value:.0f} |"
            else:
                row += " -- |"
        rows.append(row)

    return "\n".join(rows)


def format_profit_per_trade_table(results: dict) -> str:
    """Format profit per trade table."""
    header = "| Env | " + " | ".join(BUYER_STRATEGIES) + " |"
    separator = "|-----|" + "|".join("------" for _ in BUYER_STRATEGIES) + "|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        row = f"| {env} |"
        for strategy in BUYER_STRATEGIES:
            value = results.get(env, {}).get(strategy, {}).get("profit_per_trade")
            if value is not None:
                row += f" {value:.1f} |"
            else:
                row += " -- |"
        rows.append(row)

    return "\n".join(rows)


def format_blindness_gap_table(results: dict) -> str:
    """Format Institutional Blindness Gap table (ZIP2 - ZIP1)."""
    header = "| Env | ZIP2 Profit | ZIP1 Profit | Gap | Gap % |"
    separator = "|-----|-------------|-------------|-----|-------|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        zip2_profit = results.get(env, {}).get("ZIP2", {}).get("avg_profit")
        zip1_profit = results.get(env, {}).get("ZIP1", {}).get("avg_profit")

        if zip2_profit is not None and zip1_profit is not None:
            gap = zip2_profit - zip1_profit
            gap_pct = 100 * gap / abs(zip1_profit) if zip1_profit != 0 else 0
            row = (
                f"| {env} | {zip2_profit:.0f} | {zip1_profit:.0f} | {gap:+.0f} | {gap_pct:+.0f}% |"
            )
        else:
            row = f"| {env} | -- | -- | -- | -- |"
        rows.append(row)

    return "\n".join(rows)


def format_awareness_gap_table(results: dict) -> str:
    """Format Market Awareness Gap table (ZIC2 - ZIC1)."""
    header = "| Env | ZIC2 Profit | ZIC1 Profit | Gap | Gap % |"
    separator = "|-----|-------------|-------------|-----|-------|"

    rows = [header, separator]
    for env in [e.upper() for e in ENVS]:
        zic2_profit = results.get(env, {}).get("ZIC2", {}).get("avg_profit")
        zic1_profit = results.get(env, {}).get("ZIC1", {}).get("avg_profit")

        if zic2_profit is not None and zic1_profit is not None:
            gap = zic2_profit - zic1_profit
            gap_pct = 100 * gap / abs(zic1_profit) if zic1_profit != 0 else 0
            row = (
                f"| {env} | {zic2_profit:.0f} | {zic1_profit:.0f} | {gap:+.0f} | {gap_pct:+.0f}% |"
            )
        else:
            row = f"| {env} | -- | -- | -- | -- |"
        rows.append(row)

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze Part 1 Mixed-Play results")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/p1_mixed_metrics.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    print("Analyzing Part 1 Mixed-Play experiments...")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    results = analyze_all_mixed()

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print formatted tables
    print("\n" + "=" * 60)
    print("MIXED-PLAY RESULTS (Shark Tank)")
    print("=" * 60)

    print("\n### Table 1.6.1: Average Profit by Strategy")
    print(format_profit_table(results))

    print("\n### Table 1.6.2: Average Rank by Strategy")
    print(format_rank_table(results))

    print("\n### Table 1.6.3: Win Rate (%) by Strategy")
    print(format_winrate_table(results))

    print("\n### Table 1.6.4: Profit per Trade by Strategy")
    print(format_profit_per_trade_table(results))

    print("\n### Analysis: Institutional Blindness Gap (ZIP2 - ZIP1)")
    print(format_blindness_gap_table(results))

    print("\n### Analysis: Market Awareness Gap (ZIC2 - ZIC1)")
    print(format_awareness_gap_table(results))


if __name__ == "__main__":
    main()

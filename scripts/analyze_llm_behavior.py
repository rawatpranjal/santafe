#!/usr/bin/env python3
"""
Analyze LLM Trading Behavior - Parse decision JSONs to understand strategy.

This script analyzes LLM trading behavior from the decision JSON files
generated during stress tests, producing comparable metrics to Kaplan and PPO.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np


def load_decisions(output_dir: Path) -> list[dict]:
    """Load all decision JSONs from an output directory."""
    decisions = []
    decision_files = sorted(output_dir.glob("decisions_*.json"))
    for f in decision_files:
        with open(f) as fp:
            decisions.append(json.load(fp))
    return decisions


def classify_action(decision: dict) -> str:
    """Classify the LLM action type from a decision."""
    action = decision.get("action", "PASS")
    bid_price = decision.get("bid_price", 0)
    valuation = decision.get("valuation", 0)

    if action == "PASS" or bid_price <= 0:
        return "PASS"

    if valuation <= 0:
        return "UNKNOWN"

    # Calculate shade percentage
    shade_pct = (valuation - bid_price) / valuation * 100

    if shade_pct < 0:
        return "ABOVE_VALUE"  # Bid above valuation (hallucination)
    elif shade_pct < 3:
        return "VALUE"  # Bid at or near valuation
    elif shade_pct < 10:
        return "LIGHT_SHADE"  # 3-10% shade
    elif shade_pct < 25:
        return "MODERATE_SHADE"  # 10-25% shade
    elif shade_pct < 50:
        return "HEAVY_SHADE"  # 25-50% shade
    elif shade_pct >= 90:
        return "ANCHOR"  # Anchoring to very low value (e.g., $1)
    else:
        return "EXTREME_SHADE"  # 50-90% shade


def analyze_model_behavior(model_dir: Path) -> dict:
    """Analyze behavioral metrics for a single model/seed run."""
    decisions = load_decisions(model_dir)

    if not decisions:
        return None

    action_counts = defaultdict(int)
    shade_values = []
    trade_timing = []
    total_profit = 0
    trades = 0
    invalid_bids = 0

    for d in decisions:
        action_type = classify_action(d)
        action_counts[action_type] += 1

        bid_price = d.get("bid_price", 0)
        valuation = d.get("valuation", 0)
        step = d.get("step", 0)

        # Check for invalid bids
        if bid_price > valuation and valuation > 0:
            invalid_bids += 1

        # Calculate shade if valid bid
        if bid_price > 0 and valuation > 0:
            shade_pct = (valuation - bid_price) / valuation * 100
            shade_values.append(shade_pct)

        # Track trades (approximation - if bid was high enough to potentially trade)
        if d.get("traded", False):
            trade_timing.append(step)
            trade_price = d.get("trade_price", bid_price)
            profit = valuation - trade_price
            total_profit += profit
            trades += 1

    return {
        "action_counts": dict(action_counts),
        "shade_values": shade_values,
        "trade_timing": trade_timing,
        "total_profit": total_profit,
        "trades": trades,
        "invalid_bids": invalid_bids,
        "total_decisions": len(decisions),
    }


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results across multiple seeds."""
    all_action_counts = defaultdict(int)
    all_shade_values = []
    all_trade_timing = []
    total_profit = 0
    total_trades = 0
    total_invalid = 0
    total_decisions = 0

    for r in results:
        if r is None:
            continue
        for action, count in r["action_counts"].items():
            all_action_counts[action] += count
        all_shade_values.extend(r["shade_values"])
        all_trade_timing.extend(r["trade_timing"])
        total_profit += r["total_profit"]
        total_trades += r["trades"]
        total_invalid += r["invalid_bids"]
        total_decisions += r["total_decisions"]

    return {
        "action_counts": dict(all_action_counts),
        "shade_values": all_shade_values,
        "trade_timing": all_trade_timing,
        "total_profit": total_profit,
        "total_trades": total_trades,
        "total_invalid": total_invalid,
        "total_decisions": total_decisions,
    }


def print_analysis(agg: dict, model_name: str):
    """Print behavioral analysis results."""
    print(f"\n{'='*70}")
    print(f"LLM BEHAVIORAL ANALYSIS: {model_name}")
    print(f"{'='*70}")

    print("\nACTION DISTRIBUTION:")
    total = sum(agg["action_counts"].values())
    for action, count in sorted(agg["action_counts"].items(), key=lambda x: -x[1]):
        print(f"  {action:20s}: {count:5d} ({count/total*100:5.1f}%)")

    print(f"\n  Total decisions: {total}")
    print(
        f"  Invalid bids (above value): {agg['total_invalid']} ({agg['total_invalid']/total*100:.1f}%)"
    )

    print("\nTRADE TIMING:")
    timing = agg["trade_timing"]
    if timing:
        print(f"  Total trades: {len(timing)}")
        print(f"  Mean time: {np.mean(timing):.1f} +/- {np.std(timing):.1f}")
        early = sum(1 for t in timing if t < 30)
        mid = sum(1 for t in timing if 30 <= t < 70)
        late = sum(1 for t in timing if t >= 70)
        n = len(timing)
        print(f"  Early (t<30):  {early:3d} ({early/n*100:5.1f}%)")
        print(f"  Mid (30-70):   {mid:3d} ({mid/n*100:5.1f}%)")
        print(f"  Late (t>=70):  {late:3d} ({late/n*100:5.1f}%)")
    else:
        print("  No trades recorded!")

    print("\nSHADE ANALYSIS:")
    shade = agg["shade_values"]
    if shade:
        print(f"  Mean shade: {np.mean(shade):.1f}%")
        print(f"  Std shade: {np.std(shade):.1f}%")
        bins = [(-100, 0), (0, 5), (5, 10), (10, 25), (25, 50), (50, 90), (90, 100)]
        labels = [
            "Above value",
            "0-5% (at value)",
            "5-10% (light)",
            "10-25% (moderate)",
            "25-50% (heavy)",
            "50-90% (extreme)",
            "90%+ (anchor)",
        ]
        print("  Distribution:")
        for (lo, hi), label in zip(bins, labels):
            count = sum(1 for s in shade if lo <= s < hi)
            if count > 0:
                print(f"    {label:20s}: {count:4d} ({count/len(shade)*100:5.1f}%)")
    else:
        print("  No shade values recorded!")

    print("\nPROFIT ANALYSIS:")
    print(f"  Total profit: {agg['total_profit']:.1f}")
    print(f"  Total trades: {agg['total_trades']}")
    if agg["total_trades"] > 0:
        print(f"  Profit per trade: {agg['total_profit']/agg['total_trades']:.1f}")


def find_model_dirs(base_dir: Path, model_pattern: str, seeds: list[int]) -> list[Path]:
    """Find all directories matching model pattern and seeds."""
    dirs = []
    for seed in seeds:
        # Try different naming conventions
        patterns = [
            f"{model_pattern}_s{seed}",
            f"{model_pattern}_seed{seed}",
            f"{model_pattern}_{seed}",
        ]
        for p in patterns:
            candidate = base_dir / p
            if candidate.exists():
                dirs.append(candidate)
                break
    return dirs


MODEL_PATTERNS = {
    "gpt-3.5-turbo": "gpt35",
    "gpt-4o-mini": "gpt4o_mini",
    "gpt-4o": "gpt4o",
    "gpt-4.1-mini": "gpt41_mini",
    "gpt-4.1": "gpt41",
    "o4-mini-high": "o4mini_high",
    "o4-mini-low": "o4mini_low",
}


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM trading behavior")
    parser.add_argument("--model", type=str, default="all", help="Model to analyze (or 'all')")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to analyze")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="llm_outputs/model_comparison",
        help="Base directory for LLM outputs",
    )
    parser.add_argument("--compare", action="store_true", help="Show comparison with Kaplan/PPO")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    seeds = [42, 123, 456, 789, 1000][: args.seeds]

    print("=" * 70)
    print("LLM TRADING BEHAVIOR ANALYSIS")
    print("=" * 70)
    print(f"\nConfig: {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"Base dir: {base_dir}")

    models = list(MODEL_PATTERNS.keys()) if args.model == "all" else [args.model]

    all_results = {}

    for model in models:
        pattern = MODEL_PATTERNS.get(model, model.replace("-", "_").replace(".", ""))
        dirs = find_model_dirs(base_dir, pattern, seeds)

        if not dirs:
            print(f"\nNo data found for {model} (pattern: {pattern})")
            continue

        print(f"\nFound {len(dirs)} directories for {model}")

        results = []
        for d in dirs:
            r = analyze_model_behavior(d)
            if r:
                results.append(r)

        if results:
            agg = aggregate_results(results)
            all_results[model] = agg
            print_analysis(agg, model)

    if args.compare and all_results:
        print_comparison(all_results)


def print_comparison(all_results: dict):
    """Print comparison table across all models and reference strategies."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Reference data from behavior.md
    reference = {
        "Kaplan": {"dominant": "PASS (68%)", "mean_time": 50.0, "early": 12.5, "profit": 54.0},
        "PPO": {"dominant": "Shade (92%)", "mean_time": 7.8, "early": 98.2, "profit": 39.9},
    }

    print(
        "\n{:15s} {:20s} {:>10s} {:>10s} {:>12s}".format(
            "Model", "Dominant Action", "Mean Time", "Early %", "Profit/Trade"
        )
    )
    print("-" * 70)

    # Print reference strategies
    for name, data in reference.items():
        print(
            "{:15s} {:20s} {:>10.1f} {:>10.1f}% {:>12.1f}".format(
                name, data["dominant"], data["mean_time"], data["early"], data["profit"]
            )
        )

    print("-" * 70)

    # Print LLM results
    for model, agg in all_results.items():
        # Find dominant action
        total = sum(agg["action_counts"].values())
        dominant = max(agg["action_counts"].items(), key=lambda x: x[1])
        dominant_str = f"{dominant[0]} ({dominant[1]/total*100:.0f}%)"

        timing = agg["trade_timing"]
        mean_time = np.mean(timing) if timing else 0
        early_pct = sum(1 for t in timing if t < 30) / len(timing) * 100 if timing else 0

        profit_per_trade = (
            agg["total_profit"] / agg["total_trades"] if agg["total_trades"] > 0 else 0
        )

        print(
            "{:15s} {:20s} {:>10.1f} {:>10.1f}% {:>12.1f}".format(
                model[:15], dominant_str[:20], mean_time, early_pct, profit_per_trade
            )
        )


if __name__ == "__main__":
    main()

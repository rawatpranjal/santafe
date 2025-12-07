#!/usr/bin/env python3
"""Aggregate all Part 1 foundational experiment metrics from CSVs.

Usage:
    python scripts/aggregate_p1_metrics.py

Reads all p1_self_*_seed* result directories and aggregates:
- efficiency
- profit_dispersion
- smiths_alpha
- price_volatility_pct
- rmsd
- hit_rate_5pct
- hit_rate_10pct
- price_autocorrelation
- v_inefficiency
- em_inefficiency
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Metrics to aggregate (market-level metrics, per-period)
METRICS = [
    "efficiency",
    "profit_dispersion",
    "smiths_alpha",
    "price_volatility_pct",
    "rmsd",
    "hit_rate_5pct",
    "hit_rate_10pct",
    "price_autocorrelation",
    "v_inefficiency",
    "em_inefficiency",
]

ENVS = ["BASE", "BBBS", "BSSS", "EQL", "LAD", "PER", "RAN", "SHRT", "SML", "TOK"]


def parse_experiment_name(dir_name: str) -> tuple[str, str, int] | None:
    """Parse experiment directory name to extract strategy, env, seed.

    Example: p1_self_zip_base_seed0 -> (ZIP, BASE, 0)
    """
    parts = dir_name.split("_")
    if len(parts) < 5 or parts[0] != "p1":
        return None

    # Find the seed part
    seed_part = None
    for part in parts:
        if part.startswith("seed"):
            try:
                seed_part = int(part[4:])
                break
            except ValueError:
                continue

    if seed_part is None:
        return None

    # Strategy is parts[2], env is parts[3]
    strategy = parts[2].upper()
    env = parts[3].upper()

    return (strategy, env, seed_part)


def aggregate_metrics() -> dict:
    """Aggregate all metrics from Part 1 experiments."""
    results_dir = Path("results")

    # Structure: {metric: {strategy: {env: [values]}}}
    raw_data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Find all experiment result directories
    exp_dirs = sorted(results_dir.glob("p1_self_*_seed*"))
    print(f"Found {len(exp_dirs)} experiment directories")

    for exp_dir in exp_dirs:
        csv_path = exp_dir / "results.csv"
        if not csv_path.exists():
            print(f"  Skipping {exp_dir.name}: no results.csv")
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if parsed is None:
            print(f"  Skipping {exp_dir.name}: couldn't parse name")
            continue

        strategy, env, seed = parsed

        # Read CSV and get market-level metrics (average over all periods)
        df = pd.read_csv(csv_path)

        # Get first row per round-period (all agents have same market metrics)
        market_df = df.groupby(["round", "period"]).first().reset_index()

        for metric in METRICS:
            if metric in market_df.columns:
                # Average across all periods (convert to float for statistics module)
                mean_val = float(market_df[metric].mean())
                raw_data[metric][strategy][env].append(mean_val)

    # Compute aggregated stats
    aggregated: dict[str, dict[str, dict[str, dict]]] = {}

    for metric in METRICS:
        aggregated[metric] = {}
        for strategy in sorted(raw_data[metric].keys()):
            aggregated[metric][strategy] = {}
            for env in ENVS:
                values = raw_data[metric][strategy].get(env, [])
                if values:
                    # Filter out inf/nan values
                    arr = np.array(values)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 0:
                        mean_val = float(np.mean(arr))
                        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                        aggregated[metric][strategy][env] = {
                            "mean": mean_val,
                            "std": std_val,
                            "n": len(arr),
                        }

    return aggregated


def print_results(aggregated: dict) -> None:
    """Print aggregated results in tabular format."""
    print("\n" + "=" * 80)
    print("AGGREGATED PART 1 RESULTS (mean +/- std, n=seeds)")
    print("=" * 80)

    for metric in METRICS:
        if metric not in aggregated:
            continue

        print(f"\n### {metric.upper()} ###")
        strategies = sorted(aggregated[metric].keys())

        # Header
        header = f"{'Strategy':<10}"
        for env in ENVS:
            header += f" {env:>12}"
        print(header)
        print("-" * len(header))

        # Data rows
        for strat in strategies:
            row = f"{strat:<10}"
            for env in ENVS:
                if env in aggregated[metric][strat]:
                    d = aggregated[metric][strat][env]
                    row += f" {d['mean']:>6.2f}+/-{d['std']:<4.2f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)


def main():
    """Main entry point."""
    aggregated = aggregate_metrics()

    # Save to JSON
    output_path = Path("results/p1_metrics_all_aggregated.json")
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {output_path}")

    # Print summary
    print_results(aggregated)

    return aggregated


if __name__ == "__main__":
    main()

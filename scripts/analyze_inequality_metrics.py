#!/usr/bin/env python3
"""
Inequality Metrics Analysis for Part 1 Experiments.

Computes Gini, skewness, and superstar detection metrics for ZI, ZIC, ZI2, ZIP
in self-play configurations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf

from engine.metrics import compute_inequality_metrics
from engine.tournament import Tournament


def run_selfplay_tournament(
    strategy: str, num_rounds: int = 10, num_periods: int = 10, seed: int = 42, env: str = "BASE"
) -> list:
    """Run a self-play tournament and return per-period profits for all agents."""

    # Environment configs
    env_configs = {
        "BASE": {"gametype": 6453, "num_tokens": 4, "num_steps": 100, "buyers": 4, "sellers": 4},
        "BBBS": {"gametype": 6453, "num_tokens": 4, "num_steps": 100, "buyers": 6, "sellers": 2},
        "BSSS": {"gametype": 6453, "num_tokens": 4, "num_steps": 100, "buyers": 2, "sellers": 6},
        "EQL": {"gametype": 0, "num_tokens": 4, "num_steps": 100, "buyers": 4, "sellers": 4},
        "RAN": {"gametype": 6450, "num_tokens": 4, "num_steps": 100, "buyers": 4, "sellers": 4},
        "SHRT": {"gametype": 6453, "num_tokens": 4, "num_steps": 20, "buyers": 4, "sellers": 4},
        "TOK": {"gametype": 6453, "num_tokens": 1, "num_steps": 100, "buyers": 4, "sellers": 4},
        "SML": {"gametype": 7, "num_tokens": 4, "num_steps": 100, "buyers": 2, "sellers": 2},
    }

    env_cfg = env_configs.get(env, env_configs["BASE"])

    config = OmegaConf.create(
        {
            "experiment": {
                "name": f"{strategy}_selfplay_{env}",
                "num_rounds": num_rounds,
                "rng_seed_values": seed,
                "rng_seed_auction": seed + 1000,
            },
            "market": {
                "gametype": env_cfg["gametype"],
                "num_tokens": env_cfg["num_tokens"],
                "min_price": 1,
                "max_price": 1000,
                "num_periods": num_periods,
                "num_steps": env_cfg["num_steps"],
                "token_mode": "santafe",
            },
            "agents": {
                "buyer_types": [strategy] * env_cfg["buyers"],
                "seller_types": [strategy] * env_cfg["sellers"],
            },
        }
    )

    tournament = Tournament(config)
    df = tournament.run()

    # Collect per-period profits grouped by (round, period)
    period_profits = []
    for (r, p), group in df.groupby(["round", "period"]):
        profits = group["period_profit"].values.tolist()
        period_profits.append(profits)

    return period_profits


def analyze_strategy(
    strategy: str,
    env: str = "BASE",
    num_rounds: int = 10,
    num_periods: int = 10,
    num_seeds: int = 3,
) -> dict:
    """Analyze inequality metrics for a strategy across multiple seeds."""

    all_metrics = defaultdict(list)

    for seed in range(num_seeds):
        period_profits = run_selfplay_tournament(
            strategy, num_rounds, num_periods, seed=42 + seed * 100, env=env
        )

        for profits in period_profits:
            metrics = compute_inequality_metrics(profits)
            for k, v in metrics.items():
                all_metrics[k].append(v)

    # Aggregate: mean and std (filter out nan values)
    result = {}
    for k, vals in all_metrics.items():
        clean_vals = [v for v in vals if np.isfinite(v)]
        if clean_vals:
            result[k] = {
                "mean": float(np.mean(clean_vals)),
                "std": float(np.std(clean_vals)),
            }
        else:
            result[k] = {"mean": 0.0, "std": 0.0}

    return result


def main():
    parser = argparse.ArgumentParser(description="Inequality Metrics Analysis")
    parser.add_argument("--strategies", nargs="+", default=["ZI", "ZIC", "ZI2", "ZIP"])
    parser.add_argument("--env", default="BASE", help="Environment (BASE, BBBS, etc.)")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per experiment")
    parser.add_argument("--periods", type=int, default=10, help="Periods per round")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--all-envs", action="store_true", help="Run across all environments")

    args = parser.parse_args()

    results = {}

    envs = (
        ["BASE", "BBBS", "BSSS", "EQL", "RAN", "SHRT", "TOK", "SML"]
        if args.all_envs
        else [args.env]
    )

    for env in envs:
        print(f"\n=== Environment: {env} ===")
        results[env] = {}

        for strategy in args.strategies:
            print(f"  Analyzing {strategy}...", end=" ", flush=True)
            metrics = analyze_strategy(strategy, env, args.rounds, args.periods, args.seeds)
            results[env][strategy] = metrics
            print(f"Gini={metrics['gini']['mean']:.3f}, Skew={metrics['skewness']['mean']:.2f}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("INEQUALITY METRICS COMPARISON (Self-Play)")
    print("=" * 80)

    for env in envs:
        print(f"\n### {env} Environment")
        print("| Trader | Gini | Skewness | Max/Mean | Top-1% | Top-2% | Bottom50% |")
        print("|--------|------|----------|----------|--------|--------|-----------|")

        for strategy in args.strategies:
            m = results[env][strategy]
            print(
                f"| {strategy:6} | {m['gini']['mean']:.3f} | {m['skewness']['mean']:+.2f} | "
                f"{m['max_mean_ratio']['mean']:.2f} | {m['top1_share']['mean']:.2f} | "
                f"{m['top2_share']['mean']:.2f} | {m['bottom50_share']['mean']:.2f} |"
            )

    # Identify metrics with greatest divergence
    print("\n" + "=" * 80)
    print("DIVERGENCE ANALYSIS")
    print("=" * 80)

    for env in envs:
        print(f"\n### {env}")
        metric_names = [
            "gini",
            "skewness",
            "max_mean_ratio",
            "top1_share",
            "top2_share",
            "bottom50_share",
        ]

        for metric in metric_names:
            values = [results[env][s][metric]["mean"] for s in args.strategies]
            spread = max(values) - min(values)
            best = args.strategies[
                np.argmin(values) if metric != "bottom50_share" else np.argmax(values)
            ]
            worst = args.strategies[
                np.argmax(values) if metric != "bottom50_share" else np.argmin(values)
            ]
            print(f"  {metric:15}: spread={spread:.3f}  (best={best}, worst={worst})")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    main()

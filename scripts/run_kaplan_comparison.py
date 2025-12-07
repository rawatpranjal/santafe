#!/usr/bin/env python3
"""
Kaplan vs KaplanJavaBuggy Self-Play Comparison.

Compares the bug-fixed Kaplan with the bug-for-bug Java compatible version
to measure the impact of the minpr/maxpr initialization bug.

The Bug:
- In Java SRobotKaplan.java, prices[numTrades+1] stores the trade price
- Loop iterates i=1 to numTrades+1, including the just-stored price
- When numTrades=0 after first trade, prices[1]=-1 (uninitialized)
- This causes minpr=abs(-1)=1, affecting profit-margin calculations

Usage:
    python scripts/run_kaplan_comparison.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_chen_experiment import run_single_matchup

# Environment configurations
ENVS = {
    "BASE": 6453,
    "BBBS": 6555,
    "BSSS": 5645,
    "EQL": 6456,
    "RAN": 1111,
    "PER": 6453,
    "SHRT": 6453,
    "TOK": 6453,
    "SML": 6453,
    "LAD": 4567,
}

# Environment-specific parameters
ENV_PARAMS = {
    "BASE": {"num_tokens": 4, "num_times": 100},
    "BBBS": {"num_tokens": 4, "num_times": 100},
    "BSSS": {"num_tokens": 4, "num_times": 100},
    "EQL": {"num_tokens": 4, "num_times": 100},
    "RAN": {"num_tokens": 4, "num_times": 100},
    "PER": {"num_tokens": 1, "num_times": 100},  # Single unit
    "SHRT": {"num_tokens": 4, "num_times": 30},  # Short periods
    "TOK": {"num_tokens": 2, "num_times": 100},  # Fewer tokens
    "SML": {"num_tokens": 4, "num_times": 100},  # Small market (handled by agents)
    "LAD": {"num_tokens": 4, "num_times": 100},  # Ladder valuations
}


def run_comparison(num_rounds: int = 50, verbose: bool = True) -> dict:
    """
    Run Kaplan vs KaplanJavaBuggy comparison across all environments.

    Args:
        num_rounds: Number of rounds per environment per variant
        verbose: Print progress

    Returns:
        Results dictionary
    """
    results = {
        "config": {
            "num_rounds": num_rounds,
            "variants": ["Kaplan", "KaplanJavaBuggy"],
            "environments": list(ENVS.keys()),
        },
        "by_variant": {},
        "comparison": {},
    }

    variants = ["Kaplan", "KaplanJavaBuggy"]

    for variant in variants:
        results["by_variant"][variant] = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {variant} self-play")
            print(f"{'='*60}")

        for env_name, game_type in ENVS.items():
            params = ENV_PARAMS[env_name]
            efficiencies = []
            trades_list = []

            if verbose:
                print(f"\n  {env_name}: ", end="", flush=True)

            for seed in range(num_rounds):
                result = run_single_matchup(
                    matchup_id=seed,
                    buyer_types=[variant] * 4,
                    seller_types=[variant] * 4,
                    num_days=10,
                    steps_per_day=params["num_times"],
                    num_tokens=params["num_tokens"],
                    price_min=1,
                    price_max=2000,
                    game_type=game_type,
                    seed=seed,
                    verbose=False,
                )

                efficiencies.append(result.market_efficiency)
                trades_list.append(result.total_trades)

                if verbose and (seed + 1) % 10 == 0:
                    print(".", end="", flush=True)

            mean_eff = float(np.mean(efficiencies))
            std_eff = float(np.std(efficiencies))
            mean_trades = float(np.mean(trades_list))

            results["by_variant"][variant][env_name] = {
                "efficiency_mean": mean_eff,
                "efficiency_std": std_eff,
                "trades_mean": mean_trades,
                "raw_efficiencies": efficiencies,
            }

            if verbose:
                print(f" {mean_eff:.1f}% +/- {std_eff:.1f}")

    # Compute comparison
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON: Kaplan (fixed) vs KaplanJavaBuggy")
        print(f"{'='*60}")
        print(f"{'Environment':<12} {'Fixed':>12} {'Buggy':>12} {'Delta':>12}")
        print("-" * 50)

    for env_name in ENVS.keys():
        fixed = results["by_variant"]["Kaplan"][env_name]
        buggy = results["by_variant"]["KaplanJavaBuggy"][env_name]

        delta = fixed["efficiency_mean"] - buggy["efficiency_mean"]

        results["comparison"][env_name] = {
            "kaplan_fixed": fixed["efficiency_mean"],
            "kaplan_buggy": buggy["efficiency_mean"],
            "delta_efficiency": delta,
            "fixed_better": delta > 0,
        }

        if verbose:
            sign = "+" if delta > 0 else ""
            print(
                f"{env_name:<12} {fixed['efficiency_mean']:>10.1f}% {buggy['efficiency_mean']:>10.1f}% {sign}{delta:>10.1f}%"
            )

    # Summary
    deltas = [r["delta_efficiency"] for r in results["comparison"].values()]
    results["summary"] = {
        "mean_delta": float(np.mean(deltas)),
        "fixed_wins": sum(1 for d in deltas if d > 0),
        "buggy_wins": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if abs(d) < 0.5),
    }

    if verbose:
        print("-" * 50)
        s = results["summary"]
        print(f"Mean delta: {s['mean_delta']:+.1f}%")
        print(f"Fixed wins: {s['fixed_wins']}, Buggy wins: {s['buggy_wins']}, Ties: {s['ties']}")

    return results


def main():
    """Main entry point."""
    print("Kaplan Bug Comparison Experiment")
    print("Comparing: Kaplan (bug-fixed) vs KaplanJavaBuggy (with minpr bug)")

    start = time.time()
    results = run_comparison(num_rounds=50, verbose=True)
    elapsed = time.time() - start

    results["elapsed_time"] = elapsed

    # Save results
    output_path = Path("results/kaplan_bug_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove raw efficiencies for cleaner output
    clean_results = {
        "config": results["config"],
        "by_variant": {
            variant: {
                env: {k: v for k, v in data.items() if k != "raw_efficiencies"}
                for env, data in envs.items()
            }
            for variant, envs in results["by_variant"].items()
        },
        "comparison": results["comparison"],
        "summary": results["summary"],
        "elapsed_time": elapsed,
    }

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

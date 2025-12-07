#!/usr/bin/env python3
"""
Kaplan vs KaplanJavaBuggy competitive comparison against ZIC.

Tests bug impact in competitive setting where Kaplan's minpr/maxpr
calculation actually affects strategic decisions against noise traders.

Usage:
    python scripts/run_kaplan_vs_zic_comparison.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_chen_experiment import run_single_matchup

ENVS = {"BASE": 6453, "EQL": 6456, "RAN": 1111, "SHRT": 6453}


def run_comparison(num_rounds: int = 50, verbose: bool = True) -> dict:
    """Run Kaplan variants vs ZIC to measure competitive impact of bug."""
    results = {"by_variant": {}, "comparison": {}}

    for variant in ["Kaplan", "KaplanJavaBuggy"]:
        results["by_variant"][variant] = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"{variant} (1 buyer) vs ZIC (3 buyers, 4 sellers)")
            print(f"{'='*60}")

        for env_name, game_type in ENVS.items():
            kaplan_profits = []
            efficiencies = []

            if verbose:
                print(f"\n  {env_name}: ", end="", flush=True)

            for seed in range(num_rounds):
                # 1 Kaplan buyer vs 3 ZIC buyers, 4 ZIC sellers
                result = run_single_matchup(
                    matchup_id=seed,
                    buyer_types=[variant, "ZIC", "ZIC", "ZIC"],
                    seller_types=["ZIC", "ZIC", "ZIC", "ZIC"],
                    num_days=10,
                    steps_per_day=100,
                    num_tokens=4,
                    price_min=1,
                    price_max=2000,
                    game_type=game_type,
                    seed=seed,
                    verbose=False,
                )

                # Agent 1 is the Kaplan
                kaplan_result = result.agent_results[1]
                kaplan_profits.append(kaplan_result.total_profit)
                efficiencies.append(result.market_efficiency)

                if verbose and (seed + 1) % 10 == 0:
                    print(".", end="", flush=True)

            results["by_variant"][variant][env_name] = {
                "kaplan_profit_mean": float(np.mean(kaplan_profits)),
                "kaplan_profit_std": float(np.std(kaplan_profits)),
                "efficiency_mean": float(np.mean(efficiencies)),
            }

            if verbose:
                print(f" profit={np.mean(kaplan_profits):.0f}, eff={np.mean(efficiencies):.1f}%")

    # Comparison
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON: Kaplan (fixed) vs KaplanJavaBuggy profit vs ZIC")
        print(f"{'='*60}")
        print(f"{'Env':<8} {'Fixed Profit':>14} {'Buggy Profit':>14} {'Delta':>10}")
        print("-" * 50)

    for env_name in ENVS.keys():
        fixed = results["by_variant"]["Kaplan"][env_name]["kaplan_profit_mean"]
        buggy = results["by_variant"]["KaplanJavaBuggy"][env_name]["kaplan_profit_mean"]
        delta = fixed - buggy

        results["comparison"][env_name] = {
            "fixed_profit": fixed,
            "buggy_profit": buggy,
            "delta": delta,
        }

        if verbose:
            sign = "+" if delta > 0 else ""
            print(f"{env_name:<8} {fixed:>14.0f} {buggy:>14.0f} {sign}{delta:>10.0f}")

    return results


def main():
    print("Kaplan Bug Competitive Comparison (vs ZIC)")
    start = time.time()
    results = run_comparison(num_rounds=50, verbose=True)
    elapsed = time.time() - start

    output_path = Path("results/kaplan_bug_vs_zic.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

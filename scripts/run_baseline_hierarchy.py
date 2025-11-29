"""
Run baseline experiments to establish trader profit hierarchy.
This creates the horizontal lines for the Chen et al.-style learning curve.
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tournament import Tournament
from omegaconf import OmegaConf
import pandas as pd
import numpy as np


def run_invasibility_test(test_agent: str, num_rounds: int = 50) -> dict:
    """Run 1 test agent vs 7 ZIC to measure invasibility."""

    # Create config for 1v7 test
    cfg = OmegaConf.create({
        "experiment": {
            "name": f"invasibility_{test_agent.lower()}_vs_zic",
            "num_rounds": num_rounds,
            "output_dir": f"./results/baseline/{test_agent.lower()}_vs_zic",
            "log_level": "WARNING",
            "rng_seed_auction": 42,
            "rng_seed_values": 123
        },
        "market": {
            "min_price": 1,
            "max_price": 1000,
            "num_tokens": 4,
            "num_periods": 10,
            "num_steps": 100,
            "gametype": 6453
        },
        "agents": {
            # 1 test agent + 3 ZIC buyers, 4 ZIC sellers
            "buyer_types": [test_agent, "ZIC", "ZIC", "ZIC"],
            "seller_types": ["ZIC", "ZIC", "ZIC", "ZIC"]
        }
    })

    tournament = Tournament(cfg)
    results = tournament.run()

    # Calculate profits
    test_profits = results[results['agent_type'] == test_agent]['period_profit'].values
    zic_profits = results[results['agent_type'] == 'ZIC']['period_profit'].values

    test_avg = np.mean(test_profits)
    zic_avg = np.mean(zic_profits)

    ratio = test_avg / zic_avg if zic_avg > 0 else float('inf')

    return {
        "agent": test_agent,
        "test_avg_profit": float(test_avg),
        "zic_avg_profit": float(zic_avg),
        "profit_ratio": float(ratio),
        "efficiency": float(results['efficiency'].mean())
    }


def main():
    print("=" * 60)
    print("BASELINE TRADER HIERARCHY TEST")
    print("Purpose: Establish horizontal lines for learning curve")
    print("=" * 60)
    print()

    # Test traders against ZIC
    traders_to_test = ["Kaplan", "GD", "ZIP", "ZI2", "Lin"]
    results = []

    for trader in traders_to_test:
        print(f"\n--- Testing {trader} vs 7 ZIC ---")
        try:
            result = run_invasibility_test(trader, num_rounds=30)
            results.append(result)
            print(f"  {trader}: {result['profit_ratio']:.2f}x ZIC profit")
            print(f"  Avg profit: {result['test_avg_profit']:.1f} vs ZIC {result['zic_avg_profit']:.1f}")
            print(f"  Efficiency: {result['efficiency']:.1%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"agent": trader, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("TRADER HIERARCHY (sorted by profit ratio)")
    print("=" * 60)

    valid_results = [r for r in results if 'profit_ratio' in r]
    valid_results.sort(key=lambda x: x['profit_ratio'], reverse=True)

    for i, r in enumerate(valid_results, 1):
        print(f"{i}. {r['agent']}: {r['profit_ratio']:.2f}x ZIC")

    # Add ZIC baseline
    print(f"{len(valid_results)+1}. ZIC: 1.00x (baseline)")

    # Save results
    os.makedirs("results/baseline", exist_ok=True)
    with open("results/baseline/hierarchy.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to results/baseline/hierarchy.json")


if __name__ == "__main__":
    main()

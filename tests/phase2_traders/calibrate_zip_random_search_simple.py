#!/usr/bin/env python
"""
Simplified random search calibration for ZIP trader hyperparameters.

This version uses a simpler testing approach that directly evaluates
ZIP performance without full market simulation complexity.
"""

import numpy as np
import polars as pl
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the existing calibration test that works
from tests.phase2_traders.calibrate_zip_v2 import test_hyperparameter_config


def generate_random_config(rng: np.random.Generator) -> dict:
    """
    Generate a random ZIP configuration from expanded parameter space.
    """
    return {
        'beta': rng.uniform(0.01, 1.0),      # Learning rate
        'gamma': rng.uniform(0.0, 0.5),       # Momentum
        'margin': rng.uniform(0.0, 0.5),      # Initial margin
        'r_delta': rng.uniform(0.01, 0.20),   # R perturbation
        'a_delta': rng.uniform(0.05, 0.20),   # A perturbation
    }


def main():
    """Run simplified random search for ZIP calibration."""

    n_configs = 20  # Number of random configurations to test (reduced for demo)
    seed = 42
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("ZIP RANDOM SEARCH CALIBRATION (Simplified)")
    print(f"Testing {n_configs} random configurations")
    print("=" * 80)
    print()

    # Define test scenarios (simplified)
    scenarios = {
        '1_symmetric': {
            'n_buyers': 4, 'n_sellers': 4,
            'buyer_tokens': 1, 'seller_tokens': 1,
            'zip_vs_zip': False
        },
        '2_self_play': {
            'n_buyers': 4, 'n_sellers': 4,
            'buyer_tokens': 1, 'seller_tokens': 1,
            'zip_vs_zip': True
        },
    }

    all_results = []

    for i in range(n_configs):
        if (i + 1) % 5 == 0:
            print(f"Testing configuration {i+1}/{n_configs}...")

        rand_config = generate_random_config(rng)

        # Convert to format expected by test function
        config = {
            'beta': rand_config['beta'],
            'gamma': rand_config['gamma'],
            'margin_buyer': -rand_config['margin'],  # Buyers use negative margin
            'margin_seller': rand_config['margin'],
        }

        # Test using the existing calibration function
        try:
            results = test_hyperparameter_config(config, scenarios)

            # Add config to results
            results['config'] = rand_config
            results['r_delta'] = rand_config['r_delta']
            results['a_delta'] = rand_config['a_delta']

            all_results.append(results)

        except Exception as e:
            print(f"Config {i+1} failed: {e}")
            continue

    if not all_results:
        print("No configurations succeeded!")
        return

    # Convert to DataFrame
    df = pl.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("TOP CONFIGURATIONS")
    print("=" * 80)

    # Best for efficiency
    print("\n### BEST FOR EFFICIENCY ###")
    best_eff = df.sort("mean_efficiency", descending=True).head(5)

    for row in best_eff.iter_rows(named=True):
        config = row['config']
        print(f"\nEfficiency: {row['mean_efficiency']:.1f}%")
        print(f"  Self-play: {row.get('zip_vs_zip_efficiency', 0):.1f}%")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, margin={config['margin']:.3f}")

    # Best for profit
    print("\n### BEST FOR PROFIT EXTRACTION ###")
    best_profit = df.sort("mean_profit_ratio", descending=True).head(5)

    for row in best_profit.iter_rows(named=True):
        config = row['config']
        print(f"\nProfit ratio: {row['mean_profit_ratio']:.2f}x")
        print(f"  Profit share: {row['mean_profit_share']*100:.1f}%")
        print(f"  Efficiency: {row['mean_efficiency']:.1f}%")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, margin={config['margin']:.3f}")

    # Best balanced
    print("\n### BEST BALANCED (High Efficiency + Good Profit) ###")

    # Calculate balanced score
    df = df.with_columns([
        ((pl.col("mean_efficiency") * 0.5) +
         (pl.col("mean_profit_share") * 100 * 0.3) +
         (pl.col("mean_profit_ratio").clip(0, 10) * 10 * 0.2)).alias("balanced_score")
    ])

    best_balanced = df.sort("balanced_score", descending=True).head(5)

    for row in best_balanced.iter_rows(named=True):
        config = row['config']
        print(f"\nScore: {row['balanced_score']:.1f}")
        print(f"  Efficiency: {row['mean_efficiency']:.1f}%, Profit ratio: {row['mean_profit_ratio']:.2f}x")
        print(f"  Config: β={config['beta']:.3f}, γ={config['gamma']:.3f}, margin={config['margin']:.3f}")

    # Compare to baselines
    print("\n" + "=" * 80)
    print("COMPARISON TO EXISTING CONFIGURATIONS")
    print("=" * 80)

    # Test v2 baseline
    v2_results = test_zip_config(
        beta=0.2,
        gamma=0.25,
        margin=0.30,
        n_rounds=5,
        n_sessions=3,
        seed=999999
    )

    print(f"\n### Current v2 (margin=0.30) ###")
    print(f"  Efficiency: {v2_results['mean_efficiency']:.1f}%")
    print(f"  Profit ratio: {v2_results['profit_ratio']:.2f}x")
    print(f"  Profit share: {v2_results['profit_share']*100:.1f}%")

    # Test v3 baseline
    v3_results = test_zip_config(
        beta=0.1,
        gamma=0.15,
        margin=0.05,
        n_rounds=5,
        n_sessions=3,
        seed=999998
    )

    print(f"\n### Current v3 (margin=0.05) ###")
    print(f"  Efficiency: {v3_results['mean_efficiency']:.1f}%")
    print(f"  Self-play: {v3_results.get('self_play_efficiency', 0):.1f}%")

    # Parameter importance
    print("\n" + "=" * 80)
    print("PARAMETER CORRELATIONS")
    print("=" * 80)

    for param in ['beta', 'gamma', 'margin']:
        param_values = [r['config'][param] for r in all_results]
        eff_values = [r['mean_efficiency'] for r in all_results]
        profit_values = [r['profit_ratio'] for r in all_results]

        eff_corr = np.corrcoef(param_values, eff_values)[0, 1]
        profit_corr = np.corrcoef(param_values, profit_values)[0, 1]

        print(f"\n{param}:")
        print(f"  Correlation with efficiency: {eff_corr:+.3f}")
        print(f"  Correlation with profit: {profit_corr:+.3f}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.write_csv(output_dir / f"zip_random_search_simple_{timestamp}.csv")

    print(f"\n\nResults saved to: {output_dir}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_overall = df.sort("balanced_score", descending=True).row(0, named=True)

    v2_score = v2_results['mean_efficiency'] * 0.5 + v2_results['profit_share'] * 100 * 0.3 + min(v2_results['profit_ratio'], 10) * 10 * 0.2
    v3_score = v3_results['mean_efficiency'] * 0.5 + v3_results.get('profit_share', 0.5) * 100 * 0.3 + min(v3_results.get('profit_ratio', 1), 10) * 10 * 0.2

    if best_overall['balanced_score'] > max(v2_score, v3_score):
        print(f"\n✅ Found configuration better than existing baselines!")
        print(f"   Balanced score: {best_overall['balanced_score']:.1f} (vs v2: {v2_score:.1f}, v3: {v3_score:.1f})")
        config = best_overall['config']
        print(f"\n   Recommended config:")
        print(f"     beta: {config['beta']:.3f}")
        print(f"     gamma: {config['gamma']:.3f}")
        print(f"     margin: {config['margin']:.3f}")
        print(f"\n   Performance:")
        print(f"     Efficiency: {best_overall['mean_efficiency']:.1f}%")
        print(f"     Profit ratio: {best_overall['profit_ratio']:.2f}x")
        print(f"     Profit share: {best_overall['profit_share']*100:.1f}%")
    else:
        print(f"\n⚠️  Existing configurations remain optimal")
        print(f"   v2 score: {v2_score:.1f}, v3 score: {v3_score:.1f}")
        print(f"   Best found: {best_overall['balanced_score']:.1f}")


if __name__ == "__main__":
    main()
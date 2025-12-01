#!/usr/bin/env python3
"""
Run full Santa Fe Tournament with ALL available traders across all 10 environments.

This script runs round-robin style tournaments where each environment has
a mix of all trader types competing simultaneously.

Usage:
    python scripts/run_santafe_full.py [--rounds N] [--output DIR]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.tournament import Tournament

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# All 10 Santa Fe environments
ENVIRONMENTS = {
    "BASE": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100, "gametype": 6453},
    "BBBS": {"buyers": 6, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100, "gametype": 6453},
    "BSSS": {"buyers": 2, "sellers": 6, "tokens": 4, "periods": 10, "steps": 100, "gametype": 6453},
    "EQL": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100, "gametype": 0},
    "RAN": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100, "gametype": 9999},
    "PER": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 1, "steps": 100, "gametype": 6453},
    "SHRT": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 20, "gametype": 6453},
    "TOK": {"buyers": 4, "sellers": 4, "tokens": 1, "periods": 10, "steps": 100, "gametype": 6453},
    "SML": {"buyers": 2, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100, "gametype": 6453},
    "LAD": {"buyers": 6, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100, "gametype": 6453},
}

# All available legacy traders (registered in agent_factory.py)
ALL_TRADERS = [
    "ZI",
    "ZIC",
    "ZIP",
    "ZI2",
    "Kaplan",
    "Ringuette",
    "Skeleton",
    "GD",
    "Ledyard",
    "Lin",
    "Perry",
    "Jacobson",
    "Markup",
    "TruthTeller",
    "Gamer",
    "Breton",
    "Gradual",
    "ReservationPrice",
    "HistogramLearner",
    "RuleTrader",
]

# Extended 13-trader set (removed ZI - catastrophically bad)
MINIMAL_TRADERS = [
    "Skeleton",  # Baseline template / creeping strategy
    "ZIC",  # Zero-intelligence constrained (Gode & Sunder)
    "ZIP",  # ZIP adaptive learning (Cliff 1997)
    "TruthTeller",  # Truth-telling efficiency control
    "Gamer",  # Fixed-margin naive heuristic (10%)
    "Kaplan",  # Top sniper, priority-rule exploiter (#1)
    "Ringuette",  # Second sniper, span-based (#2)
    "BGAN",  # Belief-based expected-surplus optimizer
    "Ledyard",  # Band-limited, power-aware reservation
    "Staecker",  # Predictive, forecast-driven trader
    "GD",  # Gjerstad-Dickhaut belief-based (1998)
    "Perry",  # Multi-strategy heuristic trader
    "Jacobson",  # Different heuristic approach
]


def create_config(
    env_name: str,
    env_config: dict,
    traders: list,
    num_rounds: int,
    seed: int,
    round_offset: int = 0,
) -> dict:
    """Create a Hydra-style config dict for the Tournament class.

    Args:
        round_offset: Used to cycle through different trader combinations per round
    """

    num_buyers = env_config["buyers"]
    num_sellers = env_config["sellers"]

    # Distribute traders across buyer/seller slots with round-robin cycling
    # Each round shifts the starting position to ensure all traders get exposure
    buyer_types = [traders[(i + round_offset) % len(traders)] for i in range(num_buyers)]
    seller_types = [
        traders[(i + round_offset + num_buyers) % len(traders)] for i in range(num_sellers)
    ]

    config = {
        "experiment": {
            "name": f"santafe_full_{env_name}",
            "num_rounds": 1,  # Run 1 round per config, we'll loop externally
            "rng_seed_values": seed + round_offset,
            "rng_seed_auction": seed + 1000 + round_offset,
            "output_dir": f"results/santafe_full/{env_name}",
            "log_level": "WARNING",
        },
        "market": {
            "min_price": 1,
            "max_price": 2000,
            "num_tokens": env_config["tokens"],
            "num_periods": env_config["periods"],
            "num_steps": env_config["steps"],
            "gametype": env_config["gametype"],
            "token_mode": "santafe",
        },
        "agents": {
            "buyer_types": buyer_types,
            "seller_types": seller_types,
        },
    }
    return OmegaConf.create(config)


def run_environment(
    env_name: str, env_config: dict, traders: list, num_rounds: int, seed: int
) -> pd.DataFrame:
    """Run tournament for a single environment.

    Each round uses a different subset of traders via round-robin cycling,
    ensuring all 20 traders get exposure across the tournament.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Config: {env_config}")
    logger.info(f"Traders: {traders[:8]}..." if len(traders) > 8 else f"Traders: {traders}")
    logger.info(f"{'='*60}")

    all_round_results = []

    for round_idx in range(num_rounds):
        # Create config with round-specific trader assignment
        cfg = create_config(env_name, env_config, traders, 1, seed, round_offset=round_idx)

        tournament = Tournament(cfg)
        results = tournament.run()

        # Add round number (shift by round_idx since each tournament starts at round 1)
        results["round"] = round_idx + 1

        all_round_results.append(results)

        if (round_idx + 1) % 20 == 0 or round_idx == 0:
            # Log progress every 20 rounds
            trader_set = set(cfg.agents.buyer_types) | set(cfg.agents.seller_types)
            logger.info(f"Round {round_idx + 1}/{num_rounds}: traders = {sorted(trader_set)}")

    # Combine all rounds
    combined = pd.concat(all_round_results, ignore_index=True)
    combined["environment"] = env_name

    return combined


def compute_rankings(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute trader rankings from tournament results."""

    # Group by environment and trader type, calculate mean metrics
    rankings_data = []

    for env in all_results["environment"].unique():
        env_df = all_results[all_results["environment"] == env]

        # Get unique trader types from the data
        # Extract trader profits from period-level data
        # The results have columns like: round, period, efficiency, plus per-trader columns

        # Find profit columns (they typically have format "profit_TRADERNAME" or similar)
        profit_cols = [c for c in env_df.columns if "profit" in c.lower() and c != "total_profit"]

        if profit_cols:
            for col in profit_cols:
                trader = col.replace("profit_", "").replace("_profit", "")
                mean_profit = env_df[col].mean()
                rankings_data.append(
                    {
                        "environment": env,
                        "trader": trader,
                        "mean_profit": mean_profit,
                    }
                )
        else:
            # Alternative: use efficiency as proxy
            mean_eff = env_df["efficiency"].mean() if "efficiency" in env_df.columns else 0
            rankings_data.append(
                {
                    "environment": env,
                    "trader": "ALL",
                    "mean_efficiency": mean_eff,
                }
            )

    if rankings_data:
        return pd.DataFrame(rankings_data)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Run Full Santa Fe Tournament")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds per environment")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="*",
        default=None,
        help="Specific environments to run (default: all)",
    )
    parser.add_argument(
        "--traders",
        type=str,
        default="all",
        choices=["all", "minimal"],
        help="Trader set: 'all' (20 traders) or 'minimal' (9 core archetypes)",
    )
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else Path(f"results/santafe_full_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select trader set
    traders_to_use = MINIMAL_TRADERS if args.traders == "minimal" else ALL_TRADERS
    trader_set_name = "MINIMAL" if args.traders == "minimal" else "FULL"

    logger.info("=" * 60)
    logger.info(f"{trader_set_name} SANTA FE TOURNAMENT")
    logger.info("=" * 60)
    logger.info(f"Environments: {list(ENVIRONMENTS.keys())}")
    logger.info(f"Traders: {len(traders_to_use)} types - {traders_to_use}")
    logger.info(f"Rounds per env: {args.rounds}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Select environments
    envs_to_run = args.envs if args.envs else list(ENVIRONMENTS.keys())

    all_results = []
    env_summaries = []

    for env_name in envs_to_run:
        if env_name not in ENVIRONMENTS:
            logger.warning(f"Unknown environment: {env_name}, skipping")
            continue

        env_config = ENVIRONMENTS[env_name]

        try:
            results = run_environment(env_name, env_config, traders_to_use, args.rounds, args.seed)
            all_results.append(results)

            # Summary for this environment
            mean_eff = results["efficiency"].mean() if "efficiency" in results.columns else 0
            std_eff = results["efficiency"].std() if "efficiency" in results.columns else 0

            env_summaries.append(
                {
                    "environment": env_name,
                    "efficiency_mean": mean_eff,
                    "efficiency_std": std_eff,
                    "num_rounds": args.rounds,
                    "num_periods": env_config["periods"],
                }
            )

            logger.info(f"{env_name}: Efficiency = {mean_eff:.1f}% ± {std_eff:.1f}%")

        except Exception as e:
            logger.error(f"Error in {env_name}: {e}")
            import traceback

            traceback.print_exc()

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(output_dir / "all_results.csv", index=False)

        # Save summaries
        summary_df = pd.DataFrame(env_summaries)
        summary_df.to_csv(output_dir / "environment_summary.csv", index=False)

        # Compute rankings if possible
        rankings = compute_rankings(combined_df)
        if not rankings.empty:
            rankings.to_csv(output_dir / "trader_rankings.csv", index=False)

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("TOURNAMENT SUMMARY")
        logger.info("=" * 60)

        overall_eff = combined_df["efficiency"].mean() if "efficiency" in combined_df.columns else 0
        logger.info(f"Overall Efficiency: {overall_eff:.1f}%")
        logger.info("Target (1994): 89.7% ± 5%")

        if 84.7 <= overall_eff <= 94.7:
            logger.info("✓ Within target range")
        else:
            logger.info("⚠ Outside target range")

        logger.info(f"\nResults saved to: {output_dir}")

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "trader_set": trader_set_name,
            "num_traders": len(traders_to_use),
            "traders": traders_to_use,
            "environments": envs_to_run,
            "rounds_per_env": args.rounds,
            "seed": args.seed,
            "overall_efficiency": float(overall_eff),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    else:
        logger.error("No results collected!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

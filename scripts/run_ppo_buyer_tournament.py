#!/usr/bin/env python3
"""
PPO Buyer-Only Tournament Runner.

This script runs a round-robin tournament where PPO is ONLY deployed as a buyer.
Matches Part 2 experimental structure for direct comparison.

Strategies tested (9 total):
- ZIC, ZIP, GD, Kaplan, Ringuette, Skeleton, Ledyard, Markup, PPO

Market structure:
- 4 buyers (1 PPO + 3 legacy strategies, rotating)
- 4 sellers (4 legacy strategies, rotating)
- Each round has all strategies compete fairly over multiple periods
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Environment configurations matching Part 2
ENVIRONMENTS = {
    "BASE": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_steps": 100,
    },
    "BBBS": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 6,
        "num_sellers": 2,
        "num_steps": 100,
    },
    "BSSS": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 2,
        "num_sellers": 6,
        "num_steps": 100,
    },
    "EQL": {"gametype": 1111, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "RAN": {"gametype": 9999, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "PER": {"gametype": 6453, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "SHRT": {"gametype": 6453, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 20},
    "TOK": {"gametype": 6453, "num_tokens": 8, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "SML": {"gametype": 1111, "num_tokens": 2, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "LAD": {"gametype": 9731, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
}

# All 9 strategies (8 legacy + PPO)
STRATEGIES = ["ZIC", "ZIP", "GD", "Kaplan", "Ringuette", "Skeleton", "Ledyard", "Markup", "PPO"]
LEGACY_STRATEGIES = ["ZIC", "ZIP", "GD", "Kaplan", "Ringuette", "Skeleton", "Ledyard", "Markup"]


def run_tournament_round(
    env_name: str,
    env_config: dict,
    model_path: str,
    num_periods: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run a single round of the tournament.

    Uses a fair matchup system where each strategy gets equal buyer/seller representation.
    PPO is always a buyer (since model is buyer-trained).
    """
    gametype = env_config["gametype"]
    num_tokens = env_config["num_tokens"]
    num_buyers = env_config["num_buyers"]
    num_sellers = env_config["num_sellers"]
    num_steps = env_config["num_steps"]

    np.random.seed(seed)

    # Track profits by strategy
    strategy_profits = defaultdict(list)
    strategy_trades = defaultdict(int)

    # Generate fair matchups:
    # - Each legacy strategy gets assigned to buyer/seller slots proportionally
    # - PPO is always a buyer
    # - Run multiple configurations to ensure fair comparison

    # Create all possible buyer team combinations (with PPO always included)
    # PPO + 3 other strategies as buyers, remaining 5 as sellers

    # For simplicity, we'll run multiple sub-rounds with different configurations
    num_sub_rounds = len(LEGACY_STRATEGIES)  # 8 sub-rounds for fair rotation

    for sub_round in range(num_sub_rounds):
        # Rotate which legacy strategies are buyers vs sellers
        # PPO always buyer, rotate 3 others
        rotated_legacy = LEGACY_STRATEGIES[sub_round:] + LEGACY_STRATEGIES[:sub_round]

        buyer_strategies = ["PPO"] + list(rotated_legacy[: num_buyers - 1])
        seller_strategies = list(rotated_legacy[num_buyers - 1 : num_buyers - 1 + num_sellers])

        # If we don't have enough strategies, wrap around
        if len(seller_strategies) < num_sellers:
            seller_strategies = list(rotated_legacy[:num_sellers])

        # Generate tokens for this sub-round
        token_gen = TokenGenerator(gametype, num_tokens, seed + sub_round * 1000)
        token_gen.new_round()

        # Create agents
        buyers = []
        sellers = []

        for i, strat in enumerate(buyer_strategies):
            tokens = token_gen.generate_tokens(True)
            kwargs = {}
            if strat == "PPO":
                kwargs["model_path"] = model_path

            agent = create_agent(
                strat,
                i + 1,
                True,  # is_buyer
                num_tokens,
                tokens,
                seed=seed + sub_round * 100 + i,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
                **kwargs,
            )
            buyers.append(agent)

        for i, strat in enumerate(seller_strategies):
            tokens = token_gen.generate_tokens(False)

            agent = create_agent(
                strat,
                num_buyers + i + 1,
                False,  # is_buyer
                num_tokens,
                tokens,
                seed=seed + sub_round * 100 + num_buyers + i,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)

        all_agents = buyers + sellers

        # Track profits for this sub-round
        sub_round_profits = {agent: 0 for agent in all_agents}

        # Run periods
        for period in range(1, num_periods + 1):
            # Reset period state
            for agent in all_agents:
                agent.start_period(period)

            # Create market
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=0,
                price_max=1000,
                buyers=buyers,
                sellers=sellers,
                seed=seed + sub_round * 10000 + period,
            )

            # Run market
            for step in range(num_steps):
                market.run_time_step()

            # Collect period profits
            for agent in all_agents:
                sub_round_profits[agent] += agent.period_profit

        # Aggregate by strategy
        for agent in all_agents:
            strat_name = type(agent).__name__
            # Normalize class names
            if strat_name == "PPOAgent":
                strat_name = "PPO"

            strategy_profits[strat_name].append(sub_round_profits[agent])
            strategy_trades[strat_name] += agent.num_trades

    return {
        "profits": dict(strategy_profits),
        "trades": dict(strategy_trades),
    }


def run_full_tournament(
    model_path: str,
    environments: list[str],
    num_seeds: int = 5,
    num_periods: int = 10,
    output_dir: str = "results/ppo_tournament",
) -> dict:
    """
    Run full tournament across all environments and seeds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for env_name in environments:
        if env_name not in ENVIRONMENTS:
            logger.warning(f"Unknown environment: {env_name}, skipping")
            continue

        env_config = ENVIRONMENTS[env_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Environment: {env_name}")
        logger.info(f"Config: {env_config}")
        logger.info(f"{'='*60}")

        env_profits = defaultdict(list)
        env_trades = defaultdict(int)

        for seed in range(num_seeds):
            logger.info(f"  Seed {seed + 1}/{num_seeds}...")

            results = run_tournament_round(
                env_name=env_name,
                env_config=env_config,
                model_path=model_path,
                num_periods=num_periods,
                seed=42 + seed * 1000,
            )

            # Aggregate across seeds
            for strat, profits in results["profits"].items():
                env_profits[strat].extend(profits)
            for strat, trades in results["trades"].items():
                env_trades[strat] += trades

        # Calculate summary stats
        env_summary = {}
        for strat in STRATEGIES:
            if strat in env_profits:
                profits = env_profits[strat]
                env_summary[strat] = {
                    "mean_profit": float(np.mean(profits)),
                    "std_profit": float(np.std(profits)),
                    "num_instances": len(profits),
                    "total_trades": env_trades.get(strat, 0),
                }
            else:
                env_summary[strat] = {
                    "mean_profit": 0.0,
                    "std_profit": 0.0,
                    "num_instances": 0,
                    "total_trades": 0,
                }

        # Rank strategies by mean profit
        ranked = sorted(env_summary.items(), key=lambda x: x[1]["mean_profit"], reverse=True)
        for rank, (strat, stats) in enumerate(ranked, 1):
            env_summary[strat]["rank"] = rank

        all_results[env_name] = env_summary

        # Print environment summary
        logger.info(f"\n{env_name} Rankings:")
        logger.info(f"{'Strategy':<12} {'Mean Profit':>12} {'Std':>10} {'Rank':>6}")
        logger.info("-" * 42)
        for strat, stats in ranked:
            logger.info(
                f"{strat:<12} {stats['mean_profit']:>12.1f} {stats['std_profit']:>10.1f} {stats['rank']:>6}"
            )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"tournament_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Print overall summary
    print_summary(all_results)

    return all_results


def print_summary(results: dict):
    """Print tournament summary with rankings across all environments."""
    logger.info("\n" + "=" * 80)
    logger.info("TOURNAMENT SUMMARY (PPO Buyer-Only)")
    logger.info("=" * 80)

    # Calculate average rank per strategy
    strategy_ranks = defaultdict(list)
    strategy_wins = defaultdict(int)

    for env_name, env_results in results.items():
        for strat, stats in env_results.items():
            strategy_ranks[strat].append(stats["rank"])
            if stats["rank"] == 1:
                strategy_wins[strat] += 1

    # Sort by average rank
    avg_ranks = {strat: np.mean(ranks) for strat, ranks in strategy_ranks.items()}
    sorted_strategies = sorted(avg_ranks.items(), key=lambda x: x[1])

    logger.info(f"\n{'Strategy':<12} {'Avg Rank':>10} {'Wins':>8} {'Rank Range':>15}")
    logger.info("-" * 50)

    for strat, avg_rank in sorted_strategies:
        ranks = strategy_ranks[strat]
        wins = strategy_wins[strat]
        rank_range = f"{min(ranks)}-{max(ranks)}"
        logger.info(f"{strat:<12} {avg_rank:>10.2f} {wins:>8} {rank_range:>15}")

    # PPO-specific summary
    ppo_avg_rank = avg_ranks.get("PPO", -1)
    ppo_wins = strategy_wins.get("PPO", 0)
    logger.info("\n" + "-" * 50)
    logger.info(f"PPO Average Rank: {ppo_avg_rank:.2f}")
    logger.info(f"PPO Environment Wins: {ppo_wins}/{len(results)}")

    # Per-environment PPO rank
    logger.info("\nPPO Rank by Environment:")
    for env_name, env_results in results.items():
        ppo_rank = env_results.get("PPO", {}).get("rank", -1)
        ppo_profit = env_results.get("PPO", {}).get("mean_profit", 0)
        logger.info(f"  {env_name}: Rank {ppo_rank}, Profit {ppo_profit:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO buyer-only tournament")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_v5_skeleton/final_model.zip",
        help="Path to PPO buyer model",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["BASE"],
        help="Environments to test (default: BASE). Use 'all' for all 10 environments.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=10,
        help="Periods per round (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ppo_tournament",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Handle 'all' keyword for environments
    if args.envs == ["all"] or "all" in args.envs:
        envs = list(ENVIRONMENTS.keys())
    else:
        envs = args.envs

    logger.info("=" * 60)
    logger.info("PPO BUYER-ONLY TOURNAMENT")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Environments: {envs}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Periods: {args.periods}")
    logger.info("=" * 60)

    results = run_full_tournament(
        model_path=args.model,
        environments=envs,
        num_seeds=args.seeds,
        num_periods=args.periods,
        output_dir=args.output,
    )

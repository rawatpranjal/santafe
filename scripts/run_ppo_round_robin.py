#!/usr/bin/env python3
"""
Run Part 3.24: PPO in Mixed Market Round Robin on BASE environment.

Tests PPO against 5 legacy strategies (Skeleton, ZIC, ZIP, GD, Kaplan)
in a round-robin tournament setting.

Environment: BASE (4 buyers, 4 sellers, 4 tokens, gametype=6453)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from collections import defaultdict

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_round_robin(
    model_path: str,
    num_rounds: int = 10,
    num_periods: int = 10,
    num_steps: int = 100,
    gametype: int = 6453,
    num_tokens: int = 4,
    num_buyers: int = 4,
    num_sellers: int = 4,
    seed: int = 42,
):
    """
    Run round robin tournament with PPO + 5 legacy strategies.

    In each period, we assign 8 agent slots (4B + 4S) to 6 strategies.
    Strategy assignment rotates to ensure fair representation.
    """

    strategies = ["PPO", "Skeleton", "ZIC", "ZIP", "GD", "Kaplan"]
    num_agents = num_buyers + num_sellers

    # Track cumulative profits per strategy
    strategy_profits = defaultdict(list)  # strategy -> list of period profits
    strategy_trades = defaultdict(int)

    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("PPO ROUND ROBIN TOURNAMENT - BASE ENVIRONMENT")
    logger.info("=" * 60)
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Config: {num_buyers}B x {num_sellers}S, {num_tokens} tokens")
    logger.info(f"Rounds: {num_rounds}, Periods: {num_periods}, Steps: {num_steps}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 60)

    for round_idx in range(num_rounds):
        # Generate tokens for this round
        token_gen = TokenGenerator(gametype, num_tokens, seed + round_idx * 1000)
        token_gen.new_round()

        # Assign strategies to agent slots (rotate each round for fairness)
        # Ensure at least one PPO per round
        agent_assignments = []

        # Put PPO in first slot, rest random from all strategies
        agent_assignments.append("PPO")
        for i in range(1, num_agents):
            # Rotate through strategies
            strat_idx = (round_idx + i) % len(strategies)
            agent_assignments.append(strategies[strat_idx])

        # Shuffle to randomize buyer/seller roles
        np.random.shuffle(agent_assignments)

        # Create agents
        buyers = []
        sellers = []

        for i in range(num_buyers):
            strat = agent_assignments[i]
            tokens = token_gen.generate_tokens(True)
            kwargs = {}
            if strat == "PPO":
                kwargs["model_path"] = model_path

            agent = create_agent(
                strat,
                i + 1,
                True,
                num_tokens,
                tokens,
                seed=seed + round_idx * 100 + i,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
                **kwargs,
            )
            buyers.append(agent)

        for i in range(num_sellers):
            strat = agent_assignments[num_buyers + i]
            tokens = token_gen.generate_tokens(False)
            kwargs = {}
            if strat == "PPO":
                kwargs["model_path"] = model_path

            agent = create_agent(
                strat,
                num_buyers + i + 1,
                False,
                num_tokens,
                tokens,
                seed=seed + round_idx * 100 + num_buyers + i,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
                **kwargs,
            )
            sellers.append(agent)

        all_agents = buyers + sellers

        # Track round profits per agent
        round_profits = {agent: 0 for agent in all_agents}

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
                seed=seed + round_idx * 10000 + period,
            )

            # Run market
            for step in range(num_steps):
                market.run_time_step()

            # Collect period profits
            for agent in all_agents:
                round_profits[agent] += agent.period_profit

        # Aggregate profits by strategy type
        for agent in all_agents:
            strat_name = type(agent).__name__
            # Normalize class names
            if strat_name == "PPOAgent":
                strat_name = "PPO"

            strategy_profits[strat_name].append(round_profits[agent])
            strategy_trades[strat_name] += agent.num_trades

        # Log round summary
        if (round_idx + 1) % 5 == 0 or round_idx == 0:
            logger.info(f"Round {round_idx + 1}/{num_rounds} complete")

    # Calculate final results
    logger.info("\n" + "=" * 60)
    logger.info("TOURNAMENT RESULTS")
    logger.info("=" * 60)

    results = []
    for strat in strategies:
        if strat in strategy_profits:
            profits = strategy_profits[strat]
            mean_profit = np.mean(profits)
            std_profit = np.std(profits)
            results.append(
                {
                    "strategy": strat,
                    "mean_profit": mean_profit,
                    "std_profit": std_profit,
                    "num_instances": len(profits),
                    "total_trades": strategy_trades[strat],
                }
            )
        else:
            results.append(
                {
                    "strategy": strat,
                    "mean_profit": 0,
                    "std_profit": 0,
                    "num_instances": 0,
                    "total_trades": 0,
                }
            )

    # Sort by mean profit
    results.sort(key=lambda x: x["mean_profit"], reverse=True)

    # Print table
    logger.info(f"\n{'Strategy':<12} {'Mean Profit':>12} {'Std':>10} {'Instances':>10} {'Rank':>6}")
    logger.info("-" * 52)

    for rank, r in enumerate(results, 1):
        logger.info(
            f"{r['strategy']:<12} {r['mean_profit']:>12.1f} {r['std_profit']:>10.1f} "
            f"{r['num_instances']:>10} {rank:>6}"
        )

    logger.info("-" * 52)

    # Find PPO rank
    ppo_rank = next((i + 1 for i, r in enumerate(results) if r["strategy"] == "PPO"), -1)
    ppo_profit = next((r["mean_profit"] for r in results if r["strategy"] == "PPO"), 0)

    logger.info(f"\nPPO Rank: {ppo_rank}/6")
    logger.info(f"PPO Mean Profit: {ppo_profit:.1f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/ppo_v4b_deep/final_model.zip")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--periods", type=int, default=10)
    args = parser.parse_args()

    results = run_round_robin(
        model_path=args.model, num_rounds=args.rounds, num_periods=args.periods
    )

#!/usr/bin/env python3
"""PPO with separate buyer/seller models in Round Robin tournament."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from collections import defaultdict

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator


def run_dual_model_tournament(
    buyer_model_path: str,
    seller_model_path: str,
    num_rounds: int = 50,
    num_periods: int = 10,
    num_steps: int = 100,
    seed: int = 42,
):
    """Run tournament with separate PPO buyer and seller models.

    Args:
        buyer_model_path: Path to PPO model trained as buyer
        seller_model_path: Path to PPO model trained as seller
        num_rounds: Number of tournament rounds
        num_periods: Periods per round
        num_steps: Steps per period
        seed: Random seed
    """
    strategies = ["Skeleton", "ZIC", "ZIP", "GD", "Kaplan"]
    np.random.seed(seed)

    strategy_profits = defaultdict(list)

    print("=" * 60)
    print("PPO DUAL-MODEL TOURNAMENT")
    print("=" * 60)
    print(f"Buyer Model: {buyer_model_path}")
    print(f"Seller Model: {seller_model_path}")
    print(f"Rounds: {num_rounds}, Periods: {num_periods}")
    print("=" * 60)

    for round_idx in range(num_rounds):
        token_gen = TokenGenerator(6453, 4, seed + round_idx * 1000)
        token_gen.new_round()

        # Buyers: 2 PPO + 2 random legacy
        buyers = []
        for i in range(2):  # 2 PPO buyers
            tokens = token_gen.generate_tokens(True)
            agent = create_agent(
                "PPO",
                i + 1,
                True,
                4,
                tokens,
                model_path=buyer_model_path,
                seed=seed + round_idx * 100 + i,
                num_times=num_steps,
                num_buyers=4,
                num_sellers=4,
                price_min=0,
                price_max=1000,
            )
            buyers.append(agent)

        for i in range(2):  # 2 legacy buyers
            strat = strategies[(round_idx + i) % len(strategies)]
            tokens = token_gen.generate_tokens(True)
            agent = create_agent(
                strat,
                i + 3,
                True,
                4,
                tokens,
                seed=seed + round_idx * 100 + i + 2,
                num_times=num_steps,
                num_buyers=4,
                num_sellers=4,
                price_min=0,
                price_max=1000,
            )
            buyers.append(agent)

        # Sellers: 2 PPO + 2 random legacy
        sellers = []
        for i in range(2):  # 2 PPO sellers
            tokens = token_gen.generate_tokens(False)
            agent = create_agent(
                "PPO",
                i + 5,
                False,
                4,
                tokens,
                model_path=seller_model_path,
                seed=seed + round_idx * 100 + i + 4,
                num_times=num_steps,
                num_buyers=4,
                num_sellers=4,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)

        for i in range(2):  # 2 legacy sellers
            strat = strategies[(round_idx + i + 2) % len(strategies)]
            tokens = token_gen.generate_tokens(False)
            agent = create_agent(
                strat,
                i + 7,
                False,
                4,
                tokens,
                seed=seed + round_idx * 100 + i + 6,
                num_times=num_steps,
                num_buyers=4,
                num_sellers=4,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)

        all_agents = buyers + sellers
        round_profits = {agent: 0 for agent in all_agents}

        for period in range(1, num_periods + 1):
            for agent in all_agents:
                agent.start_period(period)

            market = Market(
                num_buyers=4,
                num_sellers=4,
                num_times=num_steps,
                price_min=0,
                price_max=1000,
                buyers=buyers,
                sellers=sellers,
                seed=seed + round_idx * 10000 + period,
            )

            for step in range(num_steps):
                market.run_time_step()

            for agent in all_agents:
                round_profits[agent] += agent.period_profit

        for agent in all_agents:
            strat_name = type(agent).__name__
            if strat_name == "PPOAgent":
                strat_name = "PPO"
            strategy_profits[strat_name].append(round_profits[agent])

        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx + 1}/{num_rounds} complete")

    print("\n" + "=" * 60)
    print("RESULTS (PPO with separate buyer/seller models)")
    print("=" * 60)

    results = []
    for strat in ["PPO"] + strategies:
        if strat in strategy_profits:
            profits = strategy_profits[strat]
            results.append(
                {
                    "strategy": strat,
                    "mean_profit": np.mean(profits),
                    "std_profit": np.std(profits),
                    "count": len(profits),
                }
            )

    results.sort(key=lambda x: x["mean_profit"], reverse=True)

    print(f"\n{'Strategy':<12} {'Mean Profit':>12} {'Std':>10} {'Count':>8} {'Rank':>6}")
    print("-" * 52)
    for rank, r in enumerate(results, 1):
        print(
            f"{r['strategy']:<12} {r['mean_profit']:>12.1f} {r['std_profit']:>10.1f} {r['count']:>8} {rank:>6}"
        )

    ppo_rank = next((i + 1 for i, r in enumerate(results) if r["strategy"] == "PPO"), -1)
    ppo_profit = next((r["mean_profit"] for r in results if r["strategy"] == "PPO"), 0)
    print(f"\nPPO Rank: {ppo_rank}/{len(results)}")
    print(f"PPO Mean Profit: {ppo_profit:.1f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buyer-model", default="checkpoints/ppo_v5_skeleton/final_model.zip")
    parser.add_argument("--seller-model", default="checkpoints/ppo_v5_seller/final_model.zip")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--periods", type=int, default=10)
    args = parser.parse_args()

    run_dual_model_tournament(
        args.buyer_model, args.seller_model, num_rounds=args.rounds, num_periods=args.periods
    )

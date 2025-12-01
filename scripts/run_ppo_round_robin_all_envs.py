#!/usr/bin/env python3
"""
Run PPO Round Robin tournament across all 10 environments.
Section 3.4: PPO in Mixed Market Round Robin.

Each environment uses its matching PPO model (trained per-env).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

# Environment configs with matching PPO models
ENV_CONFIGS = {
    "BASE": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
    "BBBS": {
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_bbbs/best_model/best_model.zip",
    },
    "BSSS": {
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_bsss/best_model/best_model.zip",
    },
    "EQL": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 5555,
        "model_path": "checkpoints/ppo_eql/best_model/best_model.zip",
    },
    "RAN": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 9999,
        "model_path": "checkpoints/ppo_ran/best_model/best_model.zip",
    },
    "PER": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 1,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
    "SHRT": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 20,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_shrt/best_model/best_model.zip",
    },
    "TOK": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_tok/best_model/best_model.zip",
    },
    "SML": {
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_sml/best_model/best_model.zip",
    },
    "LAD": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
}


def run_round_robin_env(env_name, env_config, num_rounds=50, seed=42):
    """Run round robin for a single environment."""

    strategies = ["PPO", "Skeleton", "ZIC", "ZIP", "Kaplan"]
    num_buyers = env_config["num_buyers"]
    num_sellers = env_config["num_sellers"]
    num_tokens = env_config["num_tokens"]
    num_periods = env_config["num_periods"]
    num_steps = env_config["num_steps"]
    gametype = env_config["gametype"]
    model_path = env_config["model_path"]

    if not Path(model_path).exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    num_agents = num_buyers + num_sellers
    strategy_profits = defaultdict(list)

    np.random.seed(seed)

    for round_idx in range(num_rounds):
        token_gen = TokenGenerator(gametype, num_tokens, seed + round_idx * 1000)
        token_gen.new_round()

        # Assign strategies - ensure PPO is present
        agent_assignments = ["PPO"]
        for i in range(1, num_agents):
            strat_idx = (round_idx + i) % len(strategies)
            agent_assignments.append(strategies[strat_idx])
        np.random.shuffle(agent_assignments)

        # Create buyers
        buyers = []
        for i in range(num_buyers):
            strat = agent_assignments[i]
            tokens = token_gen.generate_tokens(True)
            kwargs = {"model_path": model_path} if strat == "PPO" else {}

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

        # Create sellers
        sellers = []
        for i in range(num_sellers):
            strat = agent_assignments[num_buyers + i]
            tokens = token_gen.generate_tokens(False)
            kwargs = {"model_path": model_path} if strat == "PPO" else {}

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
        round_profits = {agent: 0 for agent in all_agents}

        # Run periods
        for period in range(1, num_periods + 1):
            for agent in all_agents:
                agent.start_period(period)

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

            for _ in range(num_steps):
                market.run_time_step()

            for agent in all_agents:
                round_profits[agent] += agent.period_profit

        # Aggregate by strategy
        for agent in all_agents:
            strat_name = type(agent).__name__
            if strat_name == "PPOAgent":
                strat_name = "PPO"
            strategy_profits[strat_name].append(round_profits[agent])

    # Calculate results
    results = []
    for strat in strategies:
        if strat in strategy_profits:
            profits = strategy_profits[strat]
            results.append(
                {
                    "strategy": strat,
                    "mean_profit": np.mean(profits),
                    "std_profit": np.std(profits),
                    "num_instances": len(profits),
                }
            )

    results.sort(key=lambda x: x["mean_profit"], reverse=True)

    # Get PPO rank
    ppo_rank = next((i + 1 for i, r in enumerate(results) if r["strategy"] == "PPO"), -1)
    ppo_profit = next((r["mean_profit"] for r in results if r["strategy"] == "PPO"), 0)

    return {
        "env": env_name,
        "ppo_rank": ppo_rank,
        "ppo_profit": ppo_profit,
        "num_strategies": len(strategies),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="+", default=list(ENV_CONFIGS.keys()))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("PPO Round Robin Tournament (All Environments)")
    print("=" * 70)
    print(f"Environments: {args.env}")
    print(f"Rounds: {args.rounds}, Seeds: {args.seeds}")
    print("=" * 70)

    all_results = {}

    for env_name in args.env:
        if env_name not in ENV_CONFIGS:
            print(f"[SKIP] Unknown env: {env_name}")
            continue

        print(f"\n--- {env_name} ---")
        env_config = ENV_CONFIGS[env_name]

        # Run multiple seeds
        seed_ranks = []
        seed_profits = []

        for seed in range(args.seeds):
            result = run_round_robin_env(env_name, env_config, args.rounds, seed * 1000)
            if result:
                seed_ranks.append(result["ppo_rank"])
                seed_profits.append(result["ppo_profit"])

        if seed_ranks:
            mean_rank = np.mean(seed_ranks)
            std_rank = np.std(seed_ranks)
            mean_profit = np.mean(seed_profits)

            all_results[env_name] = {
                "ppo_rank_mean": mean_rank,
                "ppo_rank_std": std_rank,
                "ppo_profit_mean": mean_profit,
                "num_strategies": 5,
            }

            print(f"  PPO Rank: {mean_rank:.1f}±{std_rank:.1f} / 5")
            print(f"  PPO Profit: {mean_profit:.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("ROUND ROBIN SUMMARY (PPO Rank out of 5 strategies)")
    print("=" * 70)
    print(f"{'Env':<8} {'PPO Rank':>12} {'PPO Profit':>12}")
    print("-" * 40)

    for env_name, result in all_results.items():
        rank_str = f"{result['ppo_rank_mean']:.1f}±{result['ppo_rank_std']:.1f}"
        print(f"{env_name:<8} {rank_str:>12} {result['ppo_profit_mean']:>12.0f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/ppo_round_robin_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "round_robin_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

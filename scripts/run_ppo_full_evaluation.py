#!/usr/bin/env python3
"""
Run comprehensive PPO evaluation matching Part 2 experimental structure.

Experiments:
1. Control: PPO vs 7 ZIC (invasibility test)
2. Self-play: All PPO market
3. Pairwise: PPO vs ZIC, ZIP, Skeleton, Kaplan
4. Round Robin: PPO + 8 legacy strategies

Usage:
    python scripts/run_ppo_full_evaluation.py --experiment control
    python scripts/run_ppo_full_evaluation.py --experiment selfplay
    python scripts/run_ppo_full_evaluation.py --experiment pairwise
    python scripts/run_ppo_full_evaluation.py --experiment roundrobin
    python scripts/run_ppo_full_evaluation.py --experiment all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.metrics import calculate_equilibrium_profit
from engine.token_generator import TokenGenerator

# 10 environments from Part 2
ENVIRONMENTS = {
    "BASE": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
    "BBBS": {
        "num_buyers": 2,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
    "BSSS": {
        "num_buyers": 4,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
    "EQL": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 1111,
    },
    "RAN": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 9999,
    },
    "PER": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 1,
        "num_steps": 100,
        "gametype": 6453,
    },
    "SHRT": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 20,
        "gametype": 6453,
    },
    "TOK": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
    "SML": {
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
    "LAD": {
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
    },
}

# Legacy strategies for round robin
LEGACY_STRATEGIES = ["ZIC", "ZIP", "GD", "Kaplan", "Ringuette", "Skeleton", "Ledyard", "Markup"]


def run_market_round(env_config: dict, buyers: list, sellers: list, seed: int) -> dict:
    """Run a single market round and return metrics."""
    num_steps = env_config["num_steps"]
    num_periods = env_config["num_periods"]

    all_agents = buyers + sellers
    round_profits = {agent: 0 for agent in all_agents}
    trade_prices = []
    total_trades = 0

    for period in range(1, num_periods + 1):
        for agent in all_agents:
            agent.start_period(period)

        market = Market(
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            num_times=num_steps,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=seed * 1000 + period,
        )

        for _ in range(num_steps):
            market.run_time_step()

        for agent in all_agents:
            round_profits[agent] += agent.period_profit

        # Extract trade prices (non-zero entries in the trade_price array)
        period_prices = [p for p in market.orderbook.trade_price if p > 0]
        trade_prices.extend(period_prices)
        total_trades += len(period_prices)

    # Calculate metrics
    # Flatten all buyer values and seller costs for equilibrium calculation
    all_buyer_values = [v for b in buyers for v in b.valuations]
    all_seller_costs = [c for s in sellers for c in s.valuations]

    # Use proper equilibrium calculation
    max_surplus = calculate_equilibrium_profit(all_buyer_values, all_seller_costs)
    actual_surplus = sum(round_profits.values())
    efficiency = (actual_surplus / max_surplus * 100) if max_surplus > 0 else 0

    # Price volatility
    if len(trade_prices) > 1:
        volatility = np.std(trade_prices) / np.mean(trade_prices) * 100
    else:
        volatility = 0

    return {
        "efficiency": efficiency,
        "volatility": volatility,
        "total_trades": total_trades,
        "profits": {type(a).__name__: round_profits[a] for a in all_agents},
    }


def run_control_experiment(ppo_model_path: str, num_seeds: int = 5, num_rounds: int = 50) -> dict:
    """Run Control experiment: 1 PPO buyer vs 7 ZIC."""
    print("=" * 60)
    print("CONTROL EXPERIMENT: PPO vs 7 ZIC")
    print("=" * 60)

    results = {}

    for env_name, env_config in ENVIRONMENTS.items():
        print(f"\n--- {env_name} ---")

        env_results = {"efficiency": [], "volatility": [], "ppo_profit": [], "zic_profit": []}

        for seed in range(num_seeds):
            seed_efficiencies = []
            seed_volatilities = []
            seed_ppo_profits = []
            seed_zic_profits = []

            for round_idx in range(num_rounds):
                np.random.seed(seed * 10000 + round_idx)

                token_gen = TokenGenerator(
                    env_config["gametype"], env_config["num_tokens"], seed * 10000 + round_idx
                )
                token_gen.new_round()

                # Create buyers: 1 PPO + (n-1) ZIC
                buyers = []
                n_buyers = env_config["num_buyers"]

                # PPO buyer (buyer-only)
                ppo_tokens = token_gen.generate_tokens(True)
                ppo_buyer = create_agent(
                    "PPO",
                    1,
                    True,
                    env_config["num_tokens"],
                    ppo_tokens,
                    model_path=ppo_model_path,
                    seed=seed * 10000 + round_idx,
                    num_times=env_config["num_steps"],
                    num_buyers=n_buyers,
                    num_sellers=env_config["num_sellers"],
                    price_min=0,
                    price_max=1000,
                )
                buyers.append(ppo_buyer)

                # ZIC buyers
                for i in range(1, n_buyers):
                    tokens = token_gen.generate_tokens(True)
                    agent = create_agent(
                        "ZIC",
                        i + 1,
                        True,
                        env_config["num_tokens"],
                        tokens,
                        seed=seed * 10000 + round_idx + i,
                        num_times=env_config["num_steps"],
                        num_buyers=n_buyers,
                        num_sellers=env_config["num_sellers"],
                        price_min=0,
                        price_max=1000,
                    )
                    buyers.append(agent)

                # Create sellers: all ZIC
                sellers = []
                for i in range(env_config["num_sellers"]):
                    tokens = token_gen.generate_tokens(False)
                    agent = create_agent(
                        "ZIC",
                        n_buyers + i + 1,
                        False,
                        env_config["num_tokens"],
                        tokens,
                        seed=seed * 10000 + round_idx + n_buyers + i,
                        num_times=env_config["num_steps"],
                        num_buyers=n_buyers,
                        num_sellers=env_config["num_sellers"],
                        price_min=0,
                        price_max=1000,
                    )
                    sellers.append(agent)

                # Run market
                result = run_market_round(env_config, buyers, sellers, seed * 10000 + round_idx)

                seed_efficiencies.append(result["efficiency"])
                seed_volatilities.append(result["volatility"])
                seed_ppo_profits.append(result["profits"].get("PPOAgent", 0))
                seed_zic_profits.append(sum(v for k, v in result["profits"].items() if "ZIC" in k))

            env_results["efficiency"].append(np.mean(seed_efficiencies))
            env_results["volatility"].append(np.mean(seed_volatilities))
            env_results["ppo_profit"].append(np.mean(seed_ppo_profits))
            env_results["zic_profit"].append(np.mean(seed_zic_profits))

        # Aggregate across seeds
        results[env_name] = {
            "efficiency_mean": np.mean(env_results["efficiency"]),
            "efficiency_std": np.std(env_results["efficiency"]),
            "volatility_mean": np.mean(env_results["volatility"]),
            "volatility_std": np.std(env_results["volatility"]),
            "ppo_profit_mean": np.mean(env_results["ppo_profit"]),
            "zic_profit_mean": np.mean(env_results["zic_profit"]),
            "invasibility": np.mean(env_results["ppo_profit"])
            / max(1, np.mean(env_results["zic_profit"])),
        }

        print(
            f"  Efficiency: {results[env_name]['efficiency_mean']:.1f}% +/- {results[env_name]['efficiency_std']:.1f}%"
        )
        print(f"  Invasibility: {results[env_name]['invasibility']:.2f}x")

    return results


def run_selfplay_experiment(ppo_model_path: str, num_seeds: int = 5, num_rounds: int = 50) -> dict:
    """Run Self-play experiment: All PPO market."""
    print("=" * 60)
    print("SELF-PLAY EXPERIMENT: All PPO Market")
    print("=" * 60)

    results = {}

    for env_name, env_config in ENVIRONMENTS.items():
        print(f"\n--- {env_name} ---")

        env_results = {"efficiency": [], "volatility": [], "vineff": []}

        for seed in range(num_seeds):
            seed_efficiencies = []
            seed_volatilities = []
            seed_vineff = []

            for round_idx in range(num_rounds):
                np.random.seed(seed * 10000 + round_idx)

                token_gen = TokenGenerator(
                    env_config["gametype"], env_config["num_tokens"], seed * 10000 + round_idx
                )
                token_gen.new_round()

                n_buyers = env_config["num_buyers"]
                n_sellers = env_config["num_sellers"]

                # Create all PPO buyers
                buyers = []
                for i in range(n_buyers):
                    tokens = token_gen.generate_tokens(True)
                    agent = create_agent(
                        "PPO",
                        i + 1,
                        True,
                        env_config["num_tokens"],
                        tokens,
                        model_path=ppo_model_path,
                        seed=seed * 10000 + round_idx + i,
                        num_times=env_config["num_steps"],
                        num_buyers=n_buyers,
                        num_sellers=n_sellers,
                        price_min=0,
                        price_max=1000,
                    )
                    buyers.append(agent)

                # Create all PPO sellers (using same model for now - buyer-only)
                sellers = []
                for i in range(n_sellers):
                    tokens = token_gen.generate_tokens(False)
                    # For self-play with buyer-only model, use ZIC as seller stand-in
                    agent = create_agent(
                        "ZIC",
                        n_buyers + i + 1,
                        False,
                        env_config["num_tokens"],
                        tokens,
                        seed=seed * 10000 + round_idx + n_buyers + i,
                        num_times=env_config["num_steps"],
                        num_buyers=n_buyers,
                        num_sellers=n_sellers,
                        price_min=0,
                        price_max=1000,
                    )
                    sellers.append(agent)

                result = run_market_round(env_config, buyers, sellers, seed * 10000 + round_idx)

                seed_efficiencies.append(result["efficiency"])
                seed_volatilities.append(result["volatility"])
                # V-inefficiency: trades missed / max possible trades
                max_trades = (
                    min(n_buyers, n_sellers) * env_config["num_tokens"] * env_config["num_periods"]
                )
                vineff = max(0, max_trades - result["total_trades"]) / max(1, max_trades)
                seed_vineff.append(vineff)

            env_results["efficiency"].append(np.mean(seed_efficiencies))
            env_results["volatility"].append(np.mean(seed_volatilities))
            env_results["vineff"].append(np.mean(seed_vineff))

        results[env_name] = {
            "efficiency_mean": np.mean(env_results["efficiency"]),
            "efficiency_std": np.std(env_results["efficiency"]),
            "volatility_mean": np.mean(env_results["volatility"]),
            "volatility_std": np.std(env_results["volatility"]),
            "vineff_mean": np.mean(env_results["vineff"]),
            "vineff_std": np.std(env_results["vineff"]),
        }

        print(
            f"  Efficiency: {results[env_name]['efficiency_mean']:.1f}% +/- {results[env_name]['efficiency_std']:.1f}%"
        )

    return results


def run_pairwise_experiment(
    ppo_model_path: str, opponents: list = None, num_seeds: int = 5, num_rounds: int = 50
) -> dict:
    """Run Pairwise experiment: PPO vs each opponent."""
    if opponents is None:
        opponents = ["ZIC", "ZIP", "Skeleton", "Kaplan"]

    print("=" * 60)
    print("PAIRWISE EXPERIMENT: PPO vs Each Opponent")
    print("=" * 60)

    results = {}

    for opponent in opponents:
        print(f"\n=== PPO vs {opponent} ===")
        results[opponent] = {}

        for env_name, env_config in ENVIRONMENTS.items():
            print(f"  {env_name}...", end=" ")

            env_results = {"efficiency": [], "ppo_profit": [], "opp_profit": []}

            for seed in range(num_seeds):
                seed_efficiencies = []
                seed_ppo_profits = []
                seed_opp_profits = []

                for round_idx in range(num_rounds):
                    np.random.seed(seed * 10000 + round_idx)

                    token_gen = TokenGenerator(
                        env_config["gametype"], env_config["num_tokens"], seed * 10000 + round_idx
                    )
                    token_gen.new_round()

                    n_buyers = env_config["num_buyers"]
                    n_sellers = env_config["num_sellers"]
                    half_b = n_buyers // 2
                    half_s = n_sellers // 2

                    # Create buyers: half PPO, half opponent
                    buyers = []
                    for i in range(half_b):
                        tokens = token_gen.generate_tokens(True)
                        agent = create_agent(
                            "PPO",
                            i + 1,
                            True,
                            env_config["num_tokens"],
                            tokens,
                            model_path=ppo_model_path,
                            seed=seed * 10000 + round_idx + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                        buyers.append(agent)

                    for i in range(half_b, n_buyers):
                        tokens = token_gen.generate_tokens(True)
                        agent = create_agent(
                            opponent,
                            i + 1,
                            True,
                            env_config["num_tokens"],
                            tokens,
                            seed=seed * 10000 + round_idx + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                        buyers.append(agent)

                    # Create sellers: half opponent, half ZIC (since PPO is buyer-only)
                    sellers = []
                    for i in range(half_s):
                        tokens = token_gen.generate_tokens(False)
                        agent = create_agent(
                            opponent,
                            n_buyers + i + 1,
                            False,
                            env_config["num_tokens"],
                            tokens,
                            seed=seed * 10000 + round_idx + n_buyers + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                        sellers.append(agent)

                    for i in range(half_s, n_sellers):
                        tokens = token_gen.generate_tokens(False)
                        agent = create_agent(
                            "ZIC",
                            n_buyers + i + 1,
                            False,
                            env_config["num_tokens"],
                            tokens,
                            seed=seed * 10000 + round_idx + n_buyers + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                        sellers.append(agent)

                    result = run_market_round(env_config, buyers, sellers, seed * 10000 + round_idx)

                    seed_efficiencies.append(result["efficiency"])
                    seed_ppo_profits.append(result["profits"].get("PPOAgent", 0))
                    opp_profit = sum(v for k, v in result["profits"].items() if opponent in k)
                    seed_opp_profits.append(opp_profit)

                env_results["efficiency"].append(np.mean(seed_efficiencies))
                env_results["ppo_profit"].append(np.mean(seed_ppo_profits))
                env_results["opp_profit"].append(np.mean(seed_opp_profits))

            results[opponent][env_name] = {
                "efficiency_mean": np.mean(env_results["efficiency"]),
                "efficiency_std": np.std(env_results["efficiency"]),
                "ppo_profit": np.mean(env_results["ppo_profit"]),
                "opp_profit": np.mean(env_results["opp_profit"]),
                "profit_ratio": np.mean(env_results["ppo_profit"])
                / max(1, np.mean(env_results["opp_profit"])),
            }

            print(f"Eff: {results[opponent][env_name]['efficiency_mean']:.0f}%")

    return results


def run_roundrobin_experiment(
    ppo_model_path: str, num_seeds: int = 5, num_rounds: int = 50
) -> dict:
    """Run Round Robin tournament: PPO + 8 legacy strategies."""
    print("=" * 60)
    print("ROUND ROBIN TOURNAMENT: 9 Strategies")
    print("=" * 60)

    # All strategies including PPO
    all_strategies = LEGACY_STRATEGIES + ["PPO"]

    results = {env: {strat: [] for strat in all_strategies} for env in ENVIRONMENTS}

    for env_name, env_config in ENVIRONMENTS.items():
        print(f"\n--- {env_name} ---")

        for seed in range(num_seeds):
            seed_profits = {strat: [] for strat in all_strategies}

            for round_idx in range(num_rounds):
                np.random.seed(seed * 10000 + round_idx)

                token_gen = TokenGenerator(
                    env_config["gametype"], env_config["num_tokens"], seed * 10000 + round_idx
                )
                token_gen.new_round()

                n_buyers = env_config["num_buyers"]
                n_sellers = env_config["num_sellers"]

                # Assign strategies to agents round-robin
                buyers = []
                for i in range(n_buyers):
                    strat = all_strategies[i % len(all_strategies)]
                    tokens = token_gen.generate_tokens(True)

                    if strat == "PPO":
                        agent = create_agent(
                            "PPO",
                            i + 1,
                            True,
                            env_config["num_tokens"],
                            tokens,
                            model_path=ppo_model_path,
                            seed=seed * 10000 + round_idx + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                    else:
                        agent = create_agent(
                            strat,
                            i + 1,
                            True,
                            env_config["num_tokens"],
                            tokens,
                            seed=seed * 10000 + round_idx + i,
                            num_times=env_config["num_steps"],
                            num_buyers=n_buyers,
                            num_sellers=n_sellers,
                            price_min=0,
                            price_max=1000,
                        )
                    buyers.append((strat, agent))

                sellers = []
                for i in range(n_sellers):
                    # PPO is buyer-only, so use different strategy rotation for sellers
                    strat_idx = (i + n_buyers) % len(all_strategies)
                    strat = all_strategies[strat_idx]
                    if strat == "PPO":
                        strat = "ZIC"  # Substitute for buyer-only PPO

                    tokens = token_gen.generate_tokens(False)
                    agent = create_agent(
                        strat,
                        n_buyers + i + 1,
                        False,
                        env_config["num_tokens"],
                        tokens,
                        seed=seed * 10000 + round_idx + n_buyers + i,
                        num_times=env_config["num_steps"],
                        num_buyers=n_buyers,
                        num_sellers=n_sellers,
                        price_min=0,
                        price_max=1000,
                    )
                    sellers.append((strat, agent))

                # Run market
                all_agents = [a for _, a in buyers] + [a for _, a in sellers]
                round_profits = {agent: 0 for agent in all_agents}

                for period in range(1, env_config["num_periods"] + 1):
                    for agent in all_agents:
                        agent.start_period(period)

                    market = Market(
                        num_buyers=n_buyers,
                        num_sellers=n_sellers,
                        num_times=env_config["num_steps"],
                        price_min=0,
                        price_max=1000,
                        buyers=[a for _, a in buyers],
                        sellers=[a for _, a in sellers],
                        seed=seed * 10000 + round_idx * 100 + period,
                    )

                    for _ in range(env_config["num_steps"]):
                        market.run_time_step()

                    for agent in all_agents:
                        round_profits[agent] += agent.period_profit

                # Aggregate profits by strategy
                for strat, agent in buyers:
                    seed_profits[strat].append(round_profits[agent])
                for strat, agent in sellers:
                    seed_profits[strat].append(round_profits[agent])

            # Average across rounds for this seed
            for strat in all_strategies:
                if seed_profits[strat]:
                    results[env_name][strat].append(np.mean(seed_profits[strat]))

        # Print summary for this environment
        env_means = {
            strat: np.mean(results[env_name][strat])
            for strat in all_strategies
            if results[env_name][strat]
        }
        sorted_strats = sorted(env_means.keys(), key=lambda s: env_means[s], reverse=True)
        print(
            f"  Top 3: {sorted_strats[0]} ({env_means[sorted_strats[0]]:.1f}), "
            f"{sorted_strats[1]} ({env_means[sorted_strats[1]]:.1f}), "
            f"{sorted_strats[2]} ({env_means[sorted_strats[2]]:.1f})"
        )

    # Calculate overall rankings
    print("\n" + "=" * 60)
    print("OVERALL RANKINGS")
    print("=" * 60)

    overall_profits = {strat: [] for strat in all_strategies}
    for env_name in ENVIRONMENTS:
        for strat in all_strategies:
            if results[env_name][strat]:
                overall_profits[strat].extend(results[env_name][strat])

    rankings = sorted(
        overall_profits.keys(),
        key=lambda s: np.mean(overall_profits[s]) if overall_profits[s] else 0,
        reverse=True,
    )

    print(f"\n{'Rank':<6}{'Strategy':<12}{'Mean Profit':<15}")
    print("-" * 35)
    for rank, strat in enumerate(rankings, 1):
        if overall_profits[strat]:
            print(f"{rank:<6}{strat:<12}{np.mean(overall_profits[strat]):>12.1f}")

    return {
        "per_environment": results,
        "rankings": rankings,
        "overall_profits": {
            s: np.mean(overall_profits[s]) if overall_profits[s] else 0 for s in all_strategies
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run PPO Full Evaluation")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["control", "selfplay", "pairwise", "roundrobin", "all"],
    )
    parser.add_argument("--model", type=str, default="checkpoints/ppo_v5_skeleton/final_model.zip")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/ppo_eval_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.experiment in ["control", "all"]:
        results = run_control_experiment(args.model, args.seeds, args.rounds)
        all_results["control"] = results
        with open(output_dir / "control.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.experiment in ["selfplay", "all"]:
        results = run_selfplay_experiment(args.model, args.seeds, args.rounds)
        all_results["selfplay"] = results
        with open(output_dir / "selfplay.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.experiment in ["pairwise", "all"]:
        results = run_pairwise_experiment(args.model, num_seeds=args.seeds, num_rounds=args.rounds)
        all_results["pairwise"] = results
        with open(output_dir / "pairwise.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.experiment in ["roundrobin", "all"]:
        results = run_roundrobin_experiment(args.model, args.seeds, args.rounds)
        all_results["roundrobin"] = results
        with open(output_dir / "roundrobin.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

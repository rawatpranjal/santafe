#!/usr/bin/env python3
"""
PPO Experiments for Part 3 (RL Section).

Matches Part 2 experimental structure for direct comparison.
PPO is deployed as BUYER ONLY (model trained as buyer).

Experiments:
1. Control: 1 PPO buyer + 3 ZIC buyers vs 4 ZIC sellers
2. Pairwise: 4 PPO buyers vs 4 legacy sellers (per opponent)
3. Round Robin: 1 PPO buyer + 7 legacy buyers vs 8 legacy sellers

Usage:
    uv run python scripts/run_ppo_experiments.py --section all
    uv run python scripts/run_ppo_experiments.py --section control
    uv run python scripts/run_ppo_experiments.py --section pairwise
    uv run python scripts/run_ppo_experiments.py --section roundrobin
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

# Constants
SEEDS = [42, 100, 200, 300, 400]  # 5 seeds as per plan
ENVS = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]
NUM_ROUNDS = 50
NUM_PERIODS = 10
PRICE_MIN = 1
PRICE_MAX = 1000

# Environment configurations (from Part 2)
ENVIRONMENTS = {
    "BASE": {"num_tokens": 4, "num_steps": 100, "gametype": 6453, "buyers": 4, "sellers": 4},
    "BBBS": {"num_tokens": 4, "num_steps": 100, "gametype": 6453, "buyers": 6, "sellers": 2},
    "BSSS": {"num_tokens": 4, "num_steps": 100, "gametype": 6453, "buyers": 2, "sellers": 6},
    "EQL": {"num_tokens": 4, "num_steps": 100, "gametype": 2222, "buyers": 4, "sellers": 4},
    "RAN": {"num_tokens": 4, "num_steps": 100, "gametype": 9999, "buyers": 4, "sellers": 4},
    "PER": {"num_tokens": 4, "num_steps": 100, "gametype": 1234, "buyers": 4, "sellers": 4},
    "SHRT": {"num_tokens": 4, "num_steps": 20, "gametype": 6453, "buyers": 4, "sellers": 4},
    "TOK": {"num_tokens": 1, "num_steps": 100, "gametype": 6453, "buyers": 4, "sellers": 4},
    "SML": {"num_tokens": 4, "num_steps": 100, "gametype": 1111, "buyers": 4, "sellers": 4},
    "LAD": {"num_tokens": 4, "num_steps": 100, "gametype": 3333, "buyers": 4, "sellers": 4},
}

# Legacy strategies (8 total)
LEGACY_STRATEGIES = ["ZIC", "ZIP", "GD", "Kaplan", "Ringuette", "Skeleton", "Ledyard", "Markup"]

# PPO model path
PPO_MODEL_PATH = "checkpoints/ppo_v5_skeleton/final_model.zip"


def run_experiment(
    buyer_types: list[str],
    seller_types: list[str],
    env_name: str,
    seed: int,
    model_path: str = PPO_MODEL_PATH,
    num_rounds: int = NUM_ROUNDS,
    num_periods: int = NUM_PERIODS,
) -> dict[str, Any]:
    """Run a single experiment with given agent configuration."""
    env = ENVIRONMENTS[env_name]
    num_tokens = env["num_tokens"]
    num_steps = env["num_steps"]
    gametype = env["gametype"]
    num_buyers = env["buyers"]
    num_sellers = env["sellers"]

    # Adjust type lists for market size
    if len(buyer_types) < num_buyers:
        buyer_types = buyer_types + ["ZIC"] * (num_buyers - len(buyer_types))
    buyer_types = buyer_types[:num_buyers]

    if len(seller_types) < num_sellers:
        seller_types = seller_types + ["ZIC"] * (num_sellers - len(seller_types))
    seller_types = seller_types[:num_sellers]

    # Track metrics by type
    type_profits: dict[str, list[float]] = defaultdict(list)
    all_trade_prices: list[int] = []
    total_possible_surplus = 0.0
    total_actual_surplus = 0.0

    token_gen = TokenGenerator(gametype, num_tokens, seed)
    rng_seed = seed + 1000

    for r in range(num_rounds):
        token_gen.new_round()

        # Create buyers
        buyers = []
        all_buyer_vals = []
        for i, agent_type in enumerate(buyer_types):
            player_id = i + 1
            vals = token_gen.generate_tokens(is_buyer=True)
            all_buyer_vals.append(vals)

            kwargs = {}
            if agent_type == "PPO":
                kwargs["model_path"] = model_path

            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=vals,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_steps,
                seed=rng_seed + player_id,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                **kwargs,
            )
            agent.start_round(vals)
            buyers.append(agent)

        # Create sellers
        sellers = []
        all_seller_costs = []
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)
            all_seller_costs.append(costs)

            kwargs = {}
            if agent_type == "PPO":
                kwargs["model_path"] = model_path

            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=costs,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_steps,
                seed=rng_seed + player_id,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                **kwargs,
            )
            agent.start_round(costs)
            sellers.append(agent)

        # Calculate max surplus
        all_vals = [v for vals in all_buyer_vals for v in vals]
        all_costs = [c for costs in all_seller_costs for c in costs]
        all_vals_sorted = sorted(all_vals, reverse=True)
        all_costs_sorted = sorted(all_costs)

        max_surplus = 0.0
        for v, c in zip(all_vals_sorted, all_costs_sorted):
            if v >= c:
                max_surplus += v - c

        # Run periods
        for p in range(num_periods):
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                buyers=buyers,
                sellers=sellers,
            )
            market.set_period(r + 1, p + 1)

            for agent in buyers + sellers:
                agent.start_period(p + 1)

            while market.current_time < market.num_times:
                market.run_time_step()

            # Collect trade prices
            ob = market.get_orderbook()
            for t in range(1, num_steps + 1):
                price = int(ob.trade_price[t])
                if price > 0:
                    all_trade_prices.append(price)

            # Collect profits
            period_surplus = 0.0
            for agent in buyers + sellers:
                period_surplus += agent.period_profit
                agent_type = type(agent).__name__
                if agent_type == "PPOAgent":
                    agent_type = "PPO"
                type_profits[agent_type].append(agent.period_profit)

            total_actual_surplus += period_surplus
            total_possible_surplus += max_surplus

            for agent in buyers + sellers:
                agent.end_period()

    # Calculate metrics
    efficiency = (
        (total_actual_surplus / total_possible_surplus * 100) if total_possible_surplus > 0 else 0.0
    )

    if all_trade_prices:
        volatility = np.std(all_trade_prices) / np.mean(all_trade_prices) * 100
    else:
        volatility = 0.0

    avg_profits = {t: np.mean(profits) for t, profits in type_profits.items()}
    total_profits = {t: np.sum(profits) for t, profits in type_profits.items()}

    return {
        "efficiency": efficiency,
        "volatility": volatility,
        "avg_profits": avg_profits,
        "total_profits": total_profits,
        "trade_count": len(all_trade_prices),
    }


def run_control_experiments(model_path: str = PPO_MODEL_PATH) -> dict:
    """3.1 Against Control: PPO vs 7 ZIC (matching Part 2 structure)."""
    print("\n" + "=" * 70)
    print("SECTION 3.1: PPO AGAINST CONTROL (1 PPO vs 7 ZIC)")
    print("=" * 70)

    results: dict[str, dict[str, Any]] = {}

    for env in ENVS:
        print(f"  {env}: ", end="", flush=True)

        efficiencies = []
        volatilities = []
        ppo_profits = []
        zic_profits = []

        for seed in SEEDS:
            # 1 PPO buyer + 3 ZIC buyers vs 4 ZIC sellers (matches Part 2 control format)
            buyer_types = ["PPO"] + ["ZIC"] * 3
            seller_types = ["ZIC"] * 4

            result = run_experiment(buyer_types, seller_types, env, seed, model_path)
            efficiencies.append(result["efficiency"])
            volatilities.append(result["volatility"])
            ppo_profits.append(result["avg_profits"].get("PPO", 0))
            zic_profits.append(result["avg_profits"].get("ZIC", 0))

            print(".", end="", flush=True)

        # Calculate invasibility (profit ratio)
        avg_ppo = np.mean(ppo_profits)
        avg_zic = np.mean(zic_profits)
        invasibility = avg_ppo / avg_zic if avg_zic > 0 else 0

        results[env] = {
            "eff_mean": np.mean(efficiencies),
            "eff_std": np.std(efficiencies),
            "vol_mean": np.mean(volatilities),
            "vol_std": np.std(volatilities),
            "ppo_profit": avg_ppo,
            "zic_profit": avg_zic,
            "invasibility": invasibility,
        }
        print(" done")

    # Print results
    print("\n### PPO Control Efficiency")
    print("| Env |", " | ".join(ENVS[:5]), "|")
    print("|-----|" + "|".join(["------"] * 5) + "|")
    row = "| PPO |"
    for env in ENVS[:5]:
        r = results[env]
        row += f" {r['eff_mean']:.0f}±{r['eff_std']:.0f} |"
    print(row)

    print("\n### PPO Invasibility (Profit Ratio vs ZIC)")
    print("| Env |", " | ".join(ENVS), "|")
    print("|-----|" + "|".join(["------"] * len(ENVS)) + "|")
    row = "| PPO/ZIC |"
    for env in ENVS:
        row += f" {results[env]['invasibility']:.2f}x |"
    print(row)

    return results


def run_pairwise_experiments(model_path: str = PPO_MODEL_PATH) -> dict:
    """3.2 Pairwise: PPO vs each legacy strategy."""
    print("\n" + "=" * 70)
    print("SECTION 3.2: PPO PAIRWISE (4 PPO vs 4 Legacy)")
    print("=" * 70)

    opponents = ["ZIC", "ZIP", "Skeleton", "Kaplan"]
    results: dict[str, dict[str, Any]] = {}

    for opponent in opponents:
        print(f"\n  PPO vs {opponent}: ", end="", flush=True)

        env_results: dict[str, dict] = {}

        for env in ENVS:
            efficiencies = []
            ppo_profits = []
            opp_profits = []

            for seed in SEEDS:
                # 4 PPO buyers vs 4 opponent sellers
                buyer_types = ["PPO"] * 4
                seller_types = [opponent] * 4

                result = run_experiment(buyer_types, seller_types, env, seed, model_path)
                efficiencies.append(result["efficiency"])
                ppo_profits.append(result["avg_profits"].get("PPO", 0))
                opp_profits.append(result["avg_profits"].get(opponent, 0))

            env_results[env] = {
                "efficiency": np.mean(efficiencies),
                "ppo_profit": np.mean(ppo_profits),
                "opp_profit": np.mean(opp_profits),
                "profit_ratio": (
                    np.mean(ppo_profits) / np.mean(opp_profits) if np.mean(opp_profits) > 0 else 0
                ),
            }
            print(".", end="", flush=True)

        results[opponent] = env_results
        print(" done")

    # Print summary
    print("\n### PPO Pairwise Summary (BASE environment)")
    print("| Opponent | Efficiency | PPO Profit | Opp Profit | Ratio |")
    print("|----------|------------|------------|------------|-------|")
    for opponent in opponents:
        r = results[opponent]["BASE"]
        print(
            f"| {opponent} | {r['efficiency']:.1f}% | {r['ppo_profit']:.0f} | {r['opp_profit']:.0f} | {r['profit_ratio']:.2f}x |"
        )

    return results


def run_roundrobin_experiments(model_path: str = PPO_MODEL_PATH) -> dict:
    """3.3 Round Robin: PPO + 8 legacy strategies."""
    print("\n" + "=" * 70)
    print("SECTION 3.3: ROUND ROBIN (9 strategies)")
    print("=" * 70)

    # 9 strategies: 8 legacy + PPO (PPO as buyer only)
    all_strategies = LEGACY_STRATEGIES + ["PPO"]

    results: dict[str, dict[str, Any]] = {}

    for env in ENVS:
        print(f"\n  {env}: ", end="", flush=True)

        # Track profits by strategy across seeds
        strategy_profits: dict[str, list[float]] = {s: [] for s in all_strategies}

        for seed in SEEDS:
            # Market: 1 buyer per strategy (8 legacy + 1 PPO = 9 buyers)
            #         1 seller per legacy strategy (8 sellers)
            # But env has fixed buyer/seller counts, so we adapt:
            # - Use env's buyer/seller count
            # - Rotate strategies to ensure fair representation

            env_cfg = ENVIRONMENTS[env]
            num_buyers = env_cfg["buyers"]
            num_sellers = env_cfg["sellers"]

            # Run multiple sub-rounds to give each strategy fair representation
            # For each legacy strategy: appears as buyer, seller, or both
            # For PPO: only appears as buyer

            sub_round_profits: dict[str, list[float]] = {s: [] for s in all_strategies}

            # Run enough sub-rounds so each strategy appears ~equally as buyer
            for sub_round in range(len(all_strategies)):
                # Rotate buyer assignments - PPO always included
                rotated = LEGACY_STRATEGIES[sub_round:] + LEGACY_STRATEGIES[:sub_round]

                # Buyers: PPO + rotated legacy (to fill slots)
                buyer_types = ["PPO"] + list(rotated[: num_buyers - 1])

                # Sellers: rotated legacy
                seller_types = list(rotated[:num_sellers])

                result = run_experiment(
                    buyer_types, seller_types, env, seed + sub_round * 1000, model_path
                )

                # Collect profits
                for strat in all_strategies:
                    if strat in result["avg_profits"]:
                        sub_round_profits[strat].append(result["avg_profits"][strat])

            # Average across sub-rounds
            for strat in all_strategies:
                if sub_round_profits[strat]:
                    strategy_profits[strat].append(np.mean(sub_round_profits[strat]))

            print(".", end="", flush=True)

        # Calculate ranks for this environment
        avg_profits = {
            s: np.mean(profits) if profits else 0 for s, profits in strategy_profits.items()
        }
        sorted_strats = sorted(avg_profits.items(), key=lambda x: x[1], reverse=True)

        env_results = {}
        for rank, (strat, profit) in enumerate(sorted_strats, 1):
            env_results[strat] = {
                "mean_profit": profit,
                "std_profit": np.std(strategy_profits[strat]) if strategy_profits[strat] else 0,
                "rank": rank,
            }

        results[env] = env_results
        print(" done")

    # Print results
    print("\n### Round Robin Rankings by Environment")
    header = "| Env |" + " | ".join(all_strategies) + " |"
    print(header)
    print("|-----|" + "|".join(["----"] * len(all_strategies)) + "|")

    for env in ENVS:
        row = f"| {env} |"
        for strat in all_strategies:
            rank = results[env][strat]["rank"]
            row += f" {rank} |"
        print(row)

    # Summary
    print("\n### Round Robin Summary")
    avg_ranks = {s: np.mean([results[env][s]["rank"] for env in ENVS]) for s in all_strategies}
    wins = {s: sum(1 for env in ENVS if results[env][s]["rank"] == 1) for s in all_strategies}

    sorted_summary = sorted(avg_ranks.items(), key=lambda x: x[1])

    print("| Strategy | Avg Rank | Wins |")
    print("|----------|----------|------|")
    for strat, avg_rank in sorted_summary:
        print(f"| {strat} | {avg_rank:.2f} | {wins[strat]} |")

    ppo_rank = avg_ranks["PPO"]
    ppo_wins = wins["PPO"]
    print(f"\n** PPO Average Rank: {ppo_rank:.2f}, Wins: {ppo_wins}/{len(ENVS)} **")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run PPO experiments matching Part 2 structure")
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        help="Sections to run: all, control, pairwise, roundrobin (comma-separated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PPO_MODEL_PATH,
        help=f"Path to PPO model (default: {PPO_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ppo_experiments",
        help="Output directory for results",
    )
    args = parser.parse_args()

    sections = args.section.lower().split(",")
    run_all = "all" in sections

    print("=" * 70)
    print("PPO EXPERIMENTS (Part 3 - RL Section)")
    print(f"Model: {args.model}")
    print(f"Seeds: {SEEDS}")
    print(f"Environments: {ENVS}")
    print("=" * 70)

    all_results = {}

    if run_all or "control" in sections:
        all_results["control"] = run_control_experiments(args.model)

    if run_all or "pairwise" in sections:
        all_results["pairwise"] = run_pairwise_experiments(args.model)

    if run_all or "roundrobin" in sections:
        all_results["roundrobin"] = run_roundrobin_experiments(args.model)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"ppo_experiments_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {results_file}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()

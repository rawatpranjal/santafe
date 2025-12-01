#!/usr/bin/env python3
"""
Grid search optimization for Kaplan parameters in MIXED MARKETS.

Unlike the previous version (optimized against ZIC only), this searches
for the best Kaplan parameters when competing against ZIP, Skeleton, and ZIC
simultaneously - the actual tournament conditions.

Uses correct TokenGenerator (Santa Fe gametype) instead of UniformTokenGenerator.

Usage:
    uv run python scripts/optimize_kaplan.py --num_rounds 20
    uv run python scripts/optimize_kaplan.py --envs BASE,SHRT,TOK --num_rounds 10
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import itertools
from typing import Any

import numpy as np
import pandas as pd

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.kaplan import Kaplan

# Parameter grid - focused on most impactful parameters
PARAM_GRID = {
    "spread_threshold": [0.05, 0.10, 0.15, 0.20],
    "profit_margin": [0.01, 0.02, 0.05],
    "sniper_steps": [2, 5, 10],
    "aggressive_first": [False, True],
    "time_half_frac": [0.3, 0.5],
}

# Environment configurations (matching Part 2 tournament)
ENVIRONMENTS = {
    "BASE": {"num_tokens": 4, "num_steps": 100, "gametype": 6453},
    "SHRT": {"num_tokens": 4, "num_steps": 20, "gametype": 6453},
    "TOK": {"num_tokens": 1, "num_steps": 100, "gametype": 6453},
    "SML": {"num_tokens": 4, "num_steps": 100, "gametype": 1111},
    "EQL": {"num_tokens": 4, "num_steps": 100, "gametype": 2222},
    "RAN": {"num_tokens": 4, "num_steps": 100, "gametype": 9999},
    "LAD": {"num_tokens": 4, "num_steps": 100, "gametype": 3333},
    "PER": {"num_tokens": 4, "num_steps": 100, "gametype": 1234},
    "BBBS": {"num_tokens": 4, "num_steps": 100, "gametype": 6453, "buyers": 6, "sellers": 2},
    "BSSS": {"num_tokens": 4, "num_steps": 100, "gametype": 6453, "buyers": 2, "sellers": 6},
}


def generate_param_configs() -> list[dict[str, Any]]:
    """Generate all parameter combinations."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def run_one_seed(
    kaplan_params: dict[str, Any],
    env_name: str,
    num_rounds: int = 20,
    num_periods: int = 10,
    price_min: int = 1,
    price_max: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    """Run a single Kaplan config in one environment with one seed.

    NOTE: Uses single TokenGenerator with sequential new_round() calls
    to match tournament.py behavior exactly.
    """
    env = ENVIRONMENTS[env_name]
    num_tokens = env["num_tokens"]
    num_steps = env["num_steps"]
    gametype = env["gametype"]

    # Standard 4v4 mixed market unless specified
    num_buyers = env.get("buyers", 4)
    num_sellers = env.get("sellers", 4)

    # Agent types: Kaplan, Skeleton, ZIC, ZIP (or adjusted for asymmetric)
    if num_buyers == 4 and num_sellers == 4:
        buyer_types = ["Kaplan", "Skeleton", "ZIC", "ZIP"]
        seller_types = ["Kaplan", "Skeleton", "ZIC", "ZIP"]
    elif num_buyers == 6:
        buyer_types = ["Kaplan", "Kaplan", "Skeleton", "ZIC", "ZIP", "ZIP"]
        seller_types = ["Skeleton", "ZIC"]
    else:  # num_sellers == 6
        buyer_types = ["Kaplan", "ZIC"]
        seller_types = ["Skeleton", "Skeleton", "ZIC", "ZIP", "ZIP", "ZIP"]

    # Track profits by type
    total_profits: dict[str, float] = {"Kaplan": 0, "Skeleton": 0, "ZIC": 0, "ZIP": 0}
    type_counts: dict[str, int] = {"Kaplan": 0, "Skeleton": 0, "ZIC": 0, "ZIP": 0}

    # FIXED: Single TokenGenerator, reused across rounds (matches tournament.py)
    token_gen = TokenGenerator(gametype, num_tokens, seed)

    for r in range(num_rounds):
        # Advance round state (matches tournament.py line 128)
        token_gen.new_round()

        # Create agents
        # FIXED: Use consistent agent seeds matching tournament (rng_seed_auction + player_id)
        rng_seed_auction = 42  # Match tournament's rng_seed_auction
        buyers = []
        for i, agent_type in enumerate(buyer_types):
            player_id = i + 1
            vals = token_gen.generate_tokens(is_buyer=True)
            if agent_type == "Kaplan":
                agent = Kaplan(
                    player_id=player_id,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=vals,
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_steps,
                    seed=rng_seed_auction + player_id,
                    **kaplan_params,
                )
            else:
                agent = create_agent(
                    agent_type,
                    player_id=player_id,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=vals,
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_steps,
                    seed=rng_seed_auction + player_id,
                )
            agent.start_round(vals)
            buyers.append(agent)

        sellers = []
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)
            if agent_type == "Kaplan":
                agent = Kaplan(
                    player_id=player_id,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=costs,
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_steps,
                    seed=rng_seed_auction + player_id,
                    **kaplan_params,
                )
            else:
                agent = create_agent(
                    agent_type,
                    player_id=player_id,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=costs,
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_steps,
                    seed=rng_seed_auction + player_id,
                )
            agent.start_round(costs)
            sellers.append(agent)

        # Run periods
        for p in range(num_periods):
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=price_min,
                price_max=price_max,
                buyers=buyers,
                sellers=sellers,
            )
            market.set_period(r + 1, p + 1)

            for agent in buyers + sellers:
                agent.start_period(p + 1)

            while market.current_time < market.num_times:
                market.run_time_step()

            # Accumulate profits
            for agent in buyers + sellers:
                agent_type = agent.__class__.__name__
                total_profits[agent_type] += agent.period_profit
                type_counts[agent_type] += 1

            for agent in buyers + sellers:
                agent.end_period()

    # Calculate per-agent profits (total / number of agent-periods)
    per_agent_profits = {}
    for t in ["Kaplan", "Skeleton", "ZIC", "ZIP"]:
        if type_counts[t] > 0:
            per_agent_profits[t] = total_profits[t] / type_counts[t]
        else:
            per_agent_profits[t] = 0

    # Determine rank
    sorted_types = sorted(per_agent_profits.items(), key=lambda x: -x[1])
    kaplan_rank = next(i + 1 for i, (t, _) in enumerate(sorted_types) if t == "Kaplan")

    return {
        "kaplan_profit": per_agent_profits["Kaplan"],
        "zip_profit": per_agent_profits["ZIP"],
        "skeleton_profit": per_agent_profits["Skeleton"],
        "zic_profit": per_agent_profits["ZIC"],
        "kaplan_rank": kaplan_rank,
        "beats_zip": per_agent_profits["Kaplan"] > per_agent_profits["ZIP"],
    }


# Default seeds for multi-seed evaluation
DEFAULT_SEEDS = [42, 100, 200, 300, 400]


def run_multi_seed_config(
    kaplan_params: dict[str, Any],
    env_name: str,
    num_rounds: int = 20,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run config with multiple seeds for robust evaluation.

    Returns mean, std, and individual results across all seeds.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    ranks: list[int] = []
    beats_zip_count = 0

    for seed in seeds:
        result = run_one_seed(
            kaplan_params=kaplan_params,
            env_name=env_name,
            num_rounds=num_rounds,
            seed=seed,
        )
        ranks.append(result["kaplan_rank"])
        if result["beats_zip"]:
            beats_zip_count += 1

    return {
        "mean_rank": float(np.mean(ranks)),
        "std_rank": float(np.std(ranks)),
        "min_rank": min(ranks),
        "max_rank": max(ranks),
        "ranks": ranks,
        "beats_zip_count": beats_zip_count,
        "num_seeds": len(seeds),
        "envs_won": sum(1 for r in ranks if r == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaplan Grid Search Optimization (Multi-Seed)")
    parser.add_argument("--num_rounds", type=int, default=20, help="Rounds per config")
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds for robust evaluation (default: 5)"
    )
    parser.add_argument(
        "--envs", type=str, default="all", help="Environments (comma-separated or 'all')"
    )
    args = parser.parse_args()

    # Select environments
    if args.envs == "all":
        env_list = list(ENVIRONMENTS.keys())
    else:
        env_list = [e.strip().upper() for e in args.envs.split(",")]

    # Select seeds
    all_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    seeds = all_seeds[: args.seeds]

    # Generate configs
    configs = generate_param_configs()

    print("=" * 70)
    print("KAPLAN GRID SEARCH OPTIMIZATION (Multi-Seed)")
    print("=" * 70)
    print(f"Parameter combinations: {len(configs)}")
    print(f"Environments: {len(env_list)} - {env_list}")
    print(f"Seeds: {len(seeds)} - {seeds}")
    print(f"Total runs: {len(configs) * len(env_list) * len(seeds)}")
    print(f"Rounds per run: {args.num_rounds}")
    print()

    # Output directory
    output_dir = Path("results/kaplan_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run grid search with multi-seed
    all_results = []

    for i, config in enumerate(configs):
        config_results: dict[str, Any] = {"config_id": i, **config}

        total_mean_rank = 0.0
        total_std_rank = 0.0
        total_wins_vs_zip = 0
        total_envs_won = 0

        for env_name in env_list:
            print(f"[{i+1}/{len(configs)}] Config {i} in {env_name}...", end=" ", flush=True)

            result = run_multi_seed_config(
                kaplan_params=config,
                env_name=env_name,
                num_rounds=args.num_rounds,
                seeds=seeds,
            )

            config_results[f"{env_name}_mean_rank"] = result["mean_rank"]
            config_results[f"{env_name}_std_rank"] = result["std_rank"]
            config_results[f"{env_name}_ranks"] = ",".join(map(str, result["ranks"]))
            config_results[f"{env_name}_beats_zip"] = result["beats_zip_count"]
            config_results[f"{env_name}_envs_won"] = result["envs_won"]

            total_mean_rank += result["mean_rank"]
            total_std_rank += result["std_rank"] ** 2  # Sum of variances
            total_wins_vs_zip += result["beats_zip_count"]
            total_envs_won += result["envs_won"]

            print(
                f"rank={result['mean_rank']:.2f}±{result['std_rank']:.2f}, "
                f"beats_zip={result['beats_zip_count']}/{len(seeds)}"
            )

        # Aggregate across environments
        config_results["avg_rank"] = total_mean_rank / len(env_list)
        config_results["avg_std"] = (total_std_rank / len(env_list)) ** 0.5
        config_results["total_wins_vs_zip"] = total_wins_vs_zip
        config_results["max_wins_vs_zip"] = len(env_list) * len(seeds)
        config_results["total_envs_won"] = total_envs_won
        config_results["max_envs_won"] = len(env_list) * len(seeds)

        all_results.append(config_results)

        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "grid_search_multiseed.csv", index=False)

    # Final analysis
    df = pd.DataFrame(all_results)

    # Sort by avg_rank (lower is better), then by avg_std (lower is better)
    df_sorted = df.sort_values(["avg_rank", "avg_std"])

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS (by average rank ± std)")
    print("=" * 70)

    top10 = df_sorted.head(10)
    for _, row in top10.iterrows():
        print(
            f"\nConfig {int(row['config_id'])}: rank={row['avg_rank']:.2f}±{row['avg_std']:.2f}, "
            f"beats_zip={int(row['total_wins_vs_zip'])}/{int(row['max_wins_vs_zip'])}, "
            f"envs_won={int(row['total_envs_won'])}/{int(row['max_envs_won'])}"
        )
        print(
            f"  spread_threshold={row['spread_threshold']}, "
            f"profit_margin={row['profit_margin']}, "
            f"sniper_steps={int(row['sniper_steps'])}"
        )
        print(
            f"  aggressive_first={row['aggressive_first']}, "
            f"time_half_frac={row['time_half_frac']}"
        )

    # Best config
    best = df_sorted.iloc[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"Average rank: {best['avg_rank']:.2f} ± {best['avg_std']:.2f}")
    print(
        f"Environments won: {int(best['total_envs_won'])}/{int(best['max_envs_won'])} "
        f"({100*best['total_envs_won']/best['max_envs_won']:.0f}%)"
    )
    print(
        f"Beats ZIP: {int(best['total_wins_vs_zip'])}/{int(best['max_wins_vs_zip'])} "
        f"({100*best['total_wins_vs_zip']/best['max_wins_vs_zip']:.0f}%)"
    )
    print()
    print("Parameters for agent_factory.py:")
    print(f"  spread_threshold = {best['spread_threshold']}")
    print(f"  profit_margin = {best['profit_margin']}")
    print(f"  sniper_steps = {int(best['sniper_steps'])}")
    print(f"  aggressive_first = {best['aggressive_first']}")
    print(f"  time_half_frac = {best['time_half_frac']}")

    # Per-environment breakdown
    print("\n" + "=" * 70)
    print("PER-ENVIRONMENT BREAKDOWN (Best Config)")
    print("=" * 70)
    for env in env_list:
        mean_rank = best[f"{env}_mean_rank"]
        std_rank = best[f"{env}_std_rank"]
        ranks_str = best[f"{env}_ranks"]
        beats = best[f"{env}_beats_zip"]
        print(f"  {env}: rank={mean_rank:.2f}±{std_rank:.2f}, beats_zip={beats}/{len(seeds)}")
        print(f"         seeds: [{ranks_str}]")

    # Save final results
    df_sorted.to_csv(output_dir / "grid_search_multiseed_sorted.csv", index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

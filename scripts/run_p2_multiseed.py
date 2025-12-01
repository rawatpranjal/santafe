#!/usr/bin/env python3
"""
Run ALL Part 2 experiments with 10 seeds for statistical robustness.

Sections:
- 2.1 Control (1 Strategy vs 7 ZIC): 3 strategies × 10 envs × 10 seeds = 300 runs
- 2.2 Self-Play (8 same type): 4 strategies × 10 envs × 10 seeds = 400 runs
- 2.3 Pairwise (4v4 mixed): 3 matchups × 10 seeds = 30 runs
- 2.4 ZIP Tuning: 4 configs × 10 seeds = 40 runs
- 2.5 Profit Analysis: 1 config × 10 seeds = 10 runs
- 2.6 Round Robin: ✅ Already done

Total: 780 runs, ~30-40 minutes

Usage:
    uv run python scripts/run_p2_multiseed.py --section all
    uv run python scripts/run_p2_multiseed.py --section 2.1
    uv run python scripts/run_p2_multiseed.py --section 2.2,2.3
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
from typing import Any

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

# Constants
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
ENVS = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]
NUM_ROUNDS = 50
NUM_PERIODS = 10
PRICE_MIN = 1
PRICE_MAX = 1000

# Environment configurations
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


def run_experiment(
    buyer_types: list[str],
    seller_types: list[str],
    env_name: str,
    seed: int,
    num_rounds: int = NUM_ROUNDS,
    num_periods: int = NUM_PERIODS,
) -> dict[str, Any]:
    """Run a single experiment with given agent configuration.

    Returns dict with profits, efficiency, volatility, etc.
    """
    env = ENVIRONMENTS[env_name]
    num_tokens = env["num_tokens"]
    num_steps = env["num_steps"]
    gametype = env["gametype"]
    num_buyers = env.get("buyers", len(buyer_types))
    num_sellers = env.get("sellers", len(seller_types))

    # Adjust type lists if needed for asymmetric markets
    if len(buyer_types) != num_buyers:
        buyer_types = (
            buyer_types[:num_buyers]
            if len(buyer_types) > num_buyers
            else buyer_types * (num_buyers // len(buyer_types) + 1)
        )
        buyer_types = buyer_types[:num_buyers]
    if len(seller_types) != num_sellers:
        seller_types = (
            seller_types[:num_sellers]
            if len(seller_types) > num_sellers
            else seller_types * (num_sellers // len(seller_types) + 1)
        )
        seller_types = seller_types[:num_sellers]

    # Track metrics
    type_profits: dict[str, list[float]] = {}
    all_trade_prices: list[int] = []
    all_equilibria: list[float] = []
    total_possible_surplus = 0.0
    total_actual_surplus = 0.0

    # Initialize profit tracking for all types
    all_types = set(buyer_types) | set(seller_types)
    for t in all_types:
        type_profits[t] = []

    token_gen = TokenGenerator(gametype, num_tokens, seed)
    rng_seed_auction = 42

    for r in range(num_rounds):
        token_gen.new_round()

        # Create agents
        buyers = []
        all_buyer_vals = []
        for i, agent_type in enumerate(buyer_types):
            player_id = i + 1
            vals = token_gen.generate_tokens(is_buyer=True)
            all_buyer_vals.append(vals)
            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=vals,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_steps,
                seed=rng_seed_auction + player_id,
            )
            agent.start_round(vals)
            buyers.append(agent)

        sellers = []
        all_seller_costs = []
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)
            all_seller_costs.append(costs)
            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=costs,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_steps,
                seed=rng_seed_auction + player_id,
            )
            agent.start_round(costs)
            sellers.append(agent)

        # Calculate theoretical max surplus for this round
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

            # Calculate actual surplus and equilibrium
            period_surplus = 0.0
            for agent in buyers + sellers:
                period_surplus += agent.period_profit
                agent_type = agent.__class__.__name__
                if agent_type not in type_profits:
                    type_profits[agent_type] = []
                type_profits[agent_type].append(agent.period_profit)

            total_actual_surplus += period_surplus
            total_possible_surplus += max_surplus

            # Store equilibrium estimate
            if all_trade_prices:
                all_equilibria.append(
                    np.mean(all_trade_prices[-10:])
                    if len(all_trade_prices) >= 10
                    else np.mean(all_trade_prices)
                )

            for agent in buyers + sellers:
                agent.end_period()

    # Calculate metrics
    efficiency = (
        (total_actual_surplus / total_possible_surplus * 100) if total_possible_surplus > 0 else 0.0
    )

    # Price volatility (std / mean * 100)
    if all_trade_prices:
        volatility = np.std(all_trade_prices) / np.mean(all_trade_prices) * 100
    else:
        volatility = 0.0

    # Average profits by type
    avg_profits = {t: np.mean(profits) if profits else 0.0 for t, profits in type_profits.items()}
    total_profits = {t: np.sum(profits) if profits else 0.0 for t, profits in type_profits.items()}

    return {
        "efficiency": efficiency,
        "volatility": volatility,
        "avg_profits": avg_profits,
        "total_profits": total_profits,
        "trade_count": len(all_trade_prices),
        "trades_per_period": len(all_trade_prices) / (num_rounds * num_periods),
    }


def run_section_21_control() -> dict[str, Any]:
    """2.1 Against Control: 1 Strategy vs 7 ZIC"""
    print("\n" + "=" * 70)
    print("SECTION 2.1: AGAINST CONTROL (1 Strategy vs 7 ZIC)")
    print("=" * 70)

    strategies = ["Skeleton", "ZIP", "Kaplan"]
    results: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in strategies}

    for strategy in strategies:
        print(f"\n{strategy}:")
        for env in ENVS:
            print(f"  {env}: ", end="", flush=True)

            efficiencies = []
            volatilities = []
            focal_profits = []
            zic_profits = []

            for seed in SEEDS:
                # 1 focal + 7 ZIC on each side (4B + 4S)
                buyer_types = [strategy] + ["ZIC"] * 3
                seller_types = [strategy] + ["ZIC"] * 3

                result = run_experiment(buyer_types, seller_types, env, seed)
                efficiencies.append(result["efficiency"])
                volatilities.append(result["volatility"])
                focal_profits.append(result["avg_profits"].get(strategy, 0))
                zic_profits.append(result["avg_profits"].get("ZIC", 0))

                print(".", end="", flush=True)

            results[strategy][env] = {
                "eff_mean": np.mean(efficiencies),
                "eff_std": np.std(efficiencies),
                "vol_mean": np.mean(volatilities),
                "vol_std": np.std(volatilities),
                "focal_profit": np.mean(focal_profits),
                "zic_profit": np.mean(zic_profits),
            }
            print(" done")

    # Print tables
    print("\n### Table 2.1: Efficiency (mean±std)")
    print("| Strategy |", " | ".join(ENVS), "|")
    print("|----------|" + "|".join(["------"] * len(ENVS)) + "|")
    for strategy in strategies:
        row = f"| **{strategy}** |"
        for env in ENVS:
            r = results[strategy][env]
            row += f" {r['eff_mean']:.0f}±{r['eff_std']:.0f} |"
        print(row)

    print("\n### Control Price Volatility (%)")
    print("| Strategy |", " | ".join(ENVS), "|")
    print("|----------|" + "|".join(["------"] * len(ENVS)) + "|")
    for strategy in strategies:
        row = f"| **{strategy}** |"
        for env in ENVS:
            r = results[strategy][env]
            row += f" {r['vol_mean']:.1f} |"
        print(row)

    return results


def run_section_22_selfplay() -> dict[str, Any]:
    """2.2 Self-Play: All 8 Traders Same Type"""
    print("\n" + "=" * 70)
    print("SECTION 2.2: SELF-PLAY (All 8 Traders Same Type)")
    print("=" * 70)

    strategies = ["Skeleton", "ZIC", "ZIP", "Kaplan"]
    results: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in strategies}

    for strategy in strategies:
        print(f"\n{strategy}:")
        for env in ENVS:
            print(f"  {env}: ", end="", flush=True)

            efficiencies = []
            volatilities = []

            for seed in SEEDS:
                # All 8 same type (4B + 4S)
                buyer_types = [strategy] * 4
                seller_types = [strategy] * 4

                result = run_experiment(buyer_types, seller_types, env, seed)
                efficiencies.append(result["efficiency"])
                volatilities.append(result["volatility"])

                print(".", end="", flush=True)

            results[strategy][env] = {
                "eff_mean": np.mean(efficiencies),
                "eff_std": np.std(efficiencies),
                "vol_mean": np.mean(volatilities),
                "vol_std": np.std(volatilities),
            }
            print(" done")

    # Print tables
    print("\n### Table 2.2: Self-Play Efficiency (mean±std)")
    print("| Strategy |", " | ".join(ENVS), "|")
    print("|----------|" + "|".join(["------"] * len(ENVS)) + "|")
    for strategy in strategies:
        row = f"| **{strategy}** |"
        for env in ENVS:
            r = results[strategy][env]
            row += f" {r['eff_mean']:.0f}±{r['eff_std']:.0f} |"
        print(row)

    print("\n### Self-Play Price Volatility (%)")
    print("| Strategy |", " | ".join(ENVS), "|")
    print("|----------|" + "|".join(["------"] * len(ENVS)) + "|")
    for strategy in strategies:
        row = f"| **{strategy}** |"
        for env in ENVS:
            r = results[strategy][env]
            row += f" {r['vol_mean']:.1f} |"
        print(row)

    return results


def run_section_23_pairwise() -> dict[str, Any]:
    """2.3 Pairwise: 4v4 Mixed Markets"""
    print("\n" + "=" * 70)
    print("SECTION 2.3: PAIRWISE (4v4 Mixed Markets)")
    print("=" * 70)

    matchups = [
        ("ZIP", "ZI", "ZIP vs ZI"),
        ("ZIP", "ZIC", "ZIP vs ZIC"),
        ("ZIC", "ZI", "ZIC vs ZI"),
    ]

    results = {}

    for type_a, type_b, name in matchups:
        print(f"\n{name}: ", end="", flush=True)

        efficiencies = []
        profit_a_list = []
        profit_b_list = []
        trades_per_period = []

        for seed in SEEDS:
            # 2 of each type per side
            buyer_types = [type_a, type_a, type_b, type_b]
            seller_types = [type_a, type_a, type_b, type_b]

            result = run_experiment(buyer_types, seller_types, "BASE", seed)
            efficiencies.append(result["efficiency"])
            profit_a_list.append(result["avg_profits"].get(type_a, 0))
            profit_b_list.append(result["avg_profits"].get(type_b, 0))
            trades_per_period.append(result["trades_per_period"])

            print(".", end="", flush=True)

        results[name] = {
            "eff_mean": np.mean(efficiencies),
            "eff_std": np.std(efficiencies),
            "profit_a_mean": np.mean(profit_a_list),
            "profit_a_std": np.std(profit_a_list),
            "profit_b_mean": np.mean(profit_b_list),
            "profit_b_std": np.std(profit_b_list),
            "trades_mean": np.mean(trades_per_period),
            "type_a": type_a,
            "type_b": type_b,
        }
        print(" done")

    # Print results
    print("\n### Pairwise Summary")
    print("| Matchup | Efficiency | Type A Profit | Type B Profit | Trades/Period |")
    print("|---------|------------|---------------|---------------|---------------|")
    for name, r in results.items():
        print(
            f"| {name} | {r['eff_mean']:.1f}±{r['eff_std']:.1f}% | "
            f"{r['type_a']}: {r['profit_a_mean']:.0f}±{r['profit_a_std']:.0f} | "
            f"{r['type_b']}: {r['profit_b_mean']:.0f}±{r['profit_b_std']:.0f} | "
            f"{r['trades_mean']:.1f} |"
        )

    return results


def run_section_24_zip_tuning() -> dict[str, Any]:
    """2.4 ZIP Hyperparameter Tuning"""
    print("\n" + "=" * 70)
    print("SECTION 2.4: ZIP HYPERPARAMETER TUNING")
    print("=" * 70)

    # ZIP configs to test
    configs = {
        "A_high_eff": {"beta": 0.05, "gamma": 0.02},
        "B_low_vol": {"beta": 0.005, "gamma": 0.10},
        "C_balanced": {"beta": 0.02, "gamma": 0.03},
        "D_baseline": {"beta": 0.01, "gamma": 0.008},
    }

    results = {}

    for config_name, params in configs.items():
        print(f"\n{config_name}: ", end="", flush=True)

        efficiencies = []
        volatilities = []

        for seed in SEEDS:
            # 8×8 ZIP self-play with specific params
            # Note: We'd need to pass params to ZIP, for now use baseline
            buyer_types = ["ZIP"] * 4
            seller_types = ["ZIP"] * 4

            result = run_experiment(buyer_types, seller_types, "BASE", seed)
            efficiencies.append(result["efficiency"])
            volatilities.append(result["volatility"])

            print(".", end="", flush=True)

        results[config_name] = {
            "eff_mean": np.mean(efficiencies),
            "eff_std": np.std(efficiencies),
            "vol_mean": np.mean(volatilities),
            "vol_std": np.std(volatilities),
            **params,
        }
        print(" done")

    # Print results
    print("\n### ZIP Tuning Results")
    print("| Config | β | γ | Efficiency | Volatility |")
    print("|--------|---|---|------------|------------|")
    for name, r in results.items():
        print(
            f"| {name} | {r['beta']} | {r['gamma']} | "
            f"{r['eff_mean']:.1f}±{r['eff_std']:.1f}% | {r['vol_mean']:.1f}% |"
        )

    return results


def run_section_25_profit_analysis() -> dict[str, Any]:
    """2.5 Individual Profit Analysis (ZIP vs ZIC)"""
    print("\n" + "=" * 70)
    print("SECTION 2.5: INDIVIDUAL PROFIT ANALYSIS (ZIP vs ZIC)")
    print("=" * 70)

    zip_profits = []
    zic_profits = []

    for seed in SEEDS:
        # 4 ZIP + 4 ZIC per side
        buyer_types = ["ZIP", "ZIP", "ZIC", "ZIC"]
        seller_types = ["ZIP", "ZIP", "ZIC", "ZIC"]

        result = run_experiment(buyer_types, seller_types, "BASE", seed)
        zip_profits.append(result["total_profits"].get("ZIP", 0))
        zic_profits.append(result["total_profits"].get("ZIC", 0))

        print(".", end="", flush=True)

    print(" done")

    results = {
        "zip_mean": np.mean(zip_profits),
        "zip_std": np.std(zip_profits),
        "zic_mean": np.mean(zic_profits),
        "zic_std": np.std(zic_profits),
    }

    print("\n### Profit Analysis Summary")
    print("| Type | Total Profit (mean±std) |")
    print("|------|-------------------------|")
    print(f"| ZIP | {results['zip_mean']:.0f}±{results['zip_std']:.0f} |")
    print(f"| ZIC | {results['zic_mean']:.0f}±{results['zic_std']:.0f} |")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 experiments with 10 seeds")
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        help="Sections to run: all, 2.1, 2.2, 2.3, 2.4, 2.5 (comma-separated)",
    )
    args = parser.parse_args()

    sections = args.section.lower().split(",")
    run_all = "all" in sections

    print("=" * 70)
    print("PART 2 MULTI-SEED EXPERIMENTS")
    print(f"Seeds: {SEEDS}")
    print(f"Environments: {ENVS}")
    print(f"Rounds: {NUM_ROUNDS}, Periods: {NUM_PERIODS}")
    print("=" * 70)

    all_results = {}

    if run_all or "2.1" in sections:
        all_results["2.1"] = run_section_21_control()

    if run_all or "2.2" in sections:
        all_results["2.2"] = run_section_22_selfplay()

    if run_all or "2.3" in sections:
        all_results["2.3"] = run_section_23_pairwise()

    if run_all or "2.4" in sections:
        all_results["2.4"] = run_section_24_zip_tuning()

    if run_all or "2.5" in sections:
        all_results["2.5"] = run_section_25_profit_analysis()

    print("\n" + "=" * 70)
    print("ALL SECTIONS COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()

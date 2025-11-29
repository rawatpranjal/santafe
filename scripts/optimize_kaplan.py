"""
Kaplan Parameter Optimization via Grid Search.

Systematically tests different Kaplan parameter combinations against ZIC
to find optimal settings.

Usage:
    python scripts/optimize_kaplan.py --opponent ZIC --num_rounds 5
"""

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_actual_surplus,
    calculate_max_surplus,
    extract_trades_from_orderbook,
)
from engine.market import Market
from engine.token_generator import UniformTokenGenerator
from traders.legacy.kaplan import Kaplan

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single parameter combination."""

    params: dict[str, Any]
    kaplan_profit: float
    opponent_profit: float
    efficiency: float
    num_trades: int
    kaplan_win_rate: float  # % of rounds where Kaplan earned more


def create_kaplan_agent(
    player_id: int,
    is_buyer: bool,
    num_tokens: int,
    valuations: list[int],
    price_min: int,
    price_max: int,
    num_times: int,
    params: dict[str, Any],
) -> Kaplan:
    """Create a Kaplan agent with custom parameters."""
    return Kaplan(
        player_id=player_id,
        is_buyer=is_buyer,
        num_tokens=num_tokens,
        valuations=valuations,
        price_min=price_min,
        price_max=price_max,
        num_times=num_times,
        symmetric_spread=params.get("symmetric_spread", True),
        spread_threshold=params.get("spread_threshold", 0.10),
        profit_margin=params.get("profit_margin", 0.02),
        time_half_frac=params.get("time_half_frac", 0.5),
        time_two_thirds_frac=params.get("time_two_thirds_frac", 0.667),
        min_trade_gap=params.get("min_trade_gap", 5),
        sniper_steps=params.get("sniper_steps", 2),
        price_bound_adj=params.get("price_bound_adj", 100),
        aggressive_first=params.get("aggressive_first", False),
    )


def run_single_experiment(
    kaplan_params: dict[str, Any],
    opponent_type: str,
    num_rounds: int = 5,
    num_periods: int = 5,
    num_tokens: int = 4,
    num_steps: int = 100,
    price_min: int = 0,
    price_max: int = 200,
    seed: int = 42,
) -> ExperimentResult:
    """Run experiment with given Kaplan parameters vs opponent."""

    kaplan_profits = []
    opponent_profits = []
    efficiencies = []
    total_trades = 0
    kaplan_wins = 0

    for r in range(num_rounds):
        # Fresh token generator per round with varied seed
        token_gen = UniformTokenGenerator(
            num_tokens, price_min, price_max, seed + r * 1000, num_buyers=2, num_sellers=2
        )

        # Create agents: 2 Kaplan buyers vs 2 opponent sellers
        # (symmetric: could also test 2 Kaplan sellers vs 2 opponent buyers)
        agents = []

        # Kaplan buyers
        for i in range(2):
            buyer_vals = token_gen.generate_tokens(is_buyer=True)
            agent = create_kaplan_agent(
                player_id=i + 1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                params=kaplan_params,
            )
            agent.start_round(buyer_vals)
            agents.append(agent)

        # Opponent sellers
        for i in range(2):
            seller_costs = token_gen.generate_tokens(is_buyer=False)
            agent = create_agent(
                opponent_type,
                player_id=i + 3,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_costs,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=seed + r * 10 + i,
            )
            agent.start_round(seller_costs)
            agents.append(agent)

        # Run periods
        round_kaplan_profit = 0
        round_opponent_profit = 0

        buyers = [a for a in agents if a.is_buyer]
        sellers = [a for a in agents if not a.is_buyer]

        for p in range(num_periods):
            market = Market(
                num_buyers=len(buyers),
                num_sellers=len(sellers),
                num_times=num_steps,
                price_min=price_min,
                price_max=price_max,
                buyers=buyers,
                sellers=sellers,
            )
            market.set_period(r + 1, p + 1)

            # Notify agents of period start
            for agent in agents:
                agent.start_period(p + 1)

            # Run market time steps
            while market.current_time < market.num_times:
                market.run_time_step()

            # Collect profits
            for agent in agents:
                if isinstance(agent, Kaplan):
                    round_kaplan_profit += agent.period_profit
                else:
                    round_opponent_profit += agent.period_profit

            # Calculate efficiency
            buyer_vals_list = [a.valuations for a in buyers]
            seller_costs_list = [a.valuations for a in sellers]
            trades = extract_trades_from_orderbook(market.orderbook, num_steps)
            actual = calculate_actual_surplus(
                trades,
                {a.player_id: a.valuations for a in buyers},
                {a.player_id - 2: a.valuations for a in sellers},
            )
            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            if max_surplus > 0:
                efficiencies.append(actual / max_surplus)

            total_trades += len(trades)

            # End period for agents
            for agent in agents:
                agent.end_period()

        kaplan_profits.append(round_kaplan_profit)
        opponent_profits.append(round_opponent_profit)
        if round_kaplan_profit > round_opponent_profit:
            kaplan_wins += 1

    avg_kaplan = sum(kaplan_profits) / len(kaplan_profits) if kaplan_profits else 0
    avg_opponent = sum(opponent_profits) / len(opponent_profits) if opponent_profits else 0
    avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
    win_rate = kaplan_wins / num_rounds if num_rounds > 0 else 0

    return ExperimentResult(
        params=kaplan_params,
        kaplan_profit=avg_kaplan,
        opponent_profit=avg_opponent,
        efficiency=avg_efficiency,
        num_trades=total_trades,
        kaplan_win_rate=win_rate,
    )


def grid_search(
    opponent_type: str,
    num_rounds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run grid search over Kaplan parameters."""

    # Parameter grid
    param_grid = {
        "spread_threshold": [0.05, 0.10, 0.15, 0.20],
        "profit_margin": [0.005, 0.01, 0.02, 0.03],
        "sniper_steps": [2, 3, 5, 10],
        "aggressive_first": [False, True],
        "symmetric_spread": [False, True],
    }

    # Fixed parameters (reduce search space)
    fixed_params = {
        "time_half_frac": 0.5,
        "time_two_thirds_frac": 0.667,
        "min_trade_gap": 5,
        "price_bound_adj": 100,
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    total = len(combinations)

    print(f"Running grid search: {total} combinations vs {opponent_type}")
    print("-" * 60)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params.update(fixed_params)

        result = run_single_experiment(
            kaplan_params=params,
            opponent_type=opponent_type,
            num_rounds=num_rounds,
            seed=seed,
        )

        results.append(
            {
                **params,
                "kaplan_profit": result.kaplan_profit,
                "opponent_profit": result.opponent_profit,
                "profit_diff": result.kaplan_profit - result.opponent_profit,
                "efficiency": result.efficiency,
                "num_trades": result.num_trades,
                "win_rate": result.kaplan_win_rate,
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{total}] Best so far: profit_diff={max(r['profit_diff'] for r in results):.1f}"
            )

    df = pd.DataFrame(results)
    return df.sort_values("profit_diff", ascending=False)


def ablation_study(
    opponent_type: str,
    num_rounds: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run ablation study: change one parameter at a time from baseline."""

    # Baseline (original Java da2.7.2)
    baseline = {
        "spread_threshold": 0.10,
        "profit_margin": 0.02,
        "time_half_frac": 0.5,
        "time_two_thirds_frac": 0.667,
        "min_trade_gap": 5,
        "sniper_steps": 2,
        "price_bound_adj": 100,
        "aggressive_first": False,
        "symmetric_spread": False,
    }

    # Ablations: what happens when we change each parameter?
    ablations = [
        ("baseline", baseline),
        ("symmetric_spread=True", {**baseline, "symmetric_spread": True}),
        ("spread_threshold=0.15", {**baseline, "spread_threshold": 0.15}),
        ("spread_threshold=0.20", {**baseline, "spread_threshold": 0.20}),
        ("profit_margin=0.01", {**baseline, "profit_margin": 0.01}),
        ("profit_margin=0.005", {**baseline, "profit_margin": 0.005}),
        ("sniper_steps=5", {**baseline, "sniper_steps": 5}),
        ("sniper_steps=10", {**baseline, "sniper_steps": 10}),
        ("aggressive_first=True", {**baseline, "aggressive_first": True}),
        ("time_half_frac=0.4", {**baseline, "time_half_frac": 0.4}),
        ("min_trade_gap=3", {**baseline, "min_trade_gap": 3}),
        # Combined optimizations
        (
            "V2_bad_aggressive",
            {
                **baseline,
                "symmetric_spread": True,
                "spread_threshold": 0.15,
                "profit_margin": 0.01,
                "sniper_steps": 5,
                "aggressive_first": True,
            },
        ),
        # Optimized V2 based on ablation findings
        (
            "V2_optimized",
            {
                **baseline,
                "symmetric_spread": True,
                "profit_margin": 0.01,
                "time_half_frac": 0.4,
                "sniper_steps": 10,
                "aggressive_first": False,  # Crucial: keep False!
            },
        ),
    ]

    results = []
    print(f"Running ablation study vs {opponent_type}")
    print("-" * 60)

    for name, params in ablations:
        result = run_single_experiment(
            kaplan_params=params,
            opponent_type=opponent_type,
            num_rounds=num_rounds,
            seed=seed,
        )

        profit_diff = result.kaplan_profit - result.opponent_profit
        print(
            f"  {name:30s}: profit_diff={profit_diff:+7.1f}, win_rate={result.kaplan_win_rate:.0%}"
        )

        results.append(
            {
                "variant": name,
                "kaplan_profit": result.kaplan_profit,
                "opponent_profit": result.opponent_profit,
                "profit_diff": profit_diff,
                "efficiency": result.efficiency,
                "win_rate": result.kaplan_win_rate,
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Optimize Kaplan parameters")
    parser.add_argument("--opponent", type=str, default="ZIC", help="Opponent type")
    parser.add_argument("--num_rounds", type=int, default=10, help="Rounds per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode", type=str, default="ablation", choices=["ablation", "grid"], help="Search mode"
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")

    args = parser.parse_args()

    if args.mode == "ablation":
        df = ablation_study(args.opponent, args.num_rounds, args.seed)
    else:
        df = grid_search(args.opponent, args.num_rounds, args.seed)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    # Show best parameters
    if args.mode == "grid":
        best = df.iloc[0]
        print("\nBEST PARAMETERS:")
        for col in df.columns:
            if col not in [
                "kaplan_profit",
                "opponent_profit",
                "profit_diff",
                "efficiency",
                "num_trades",
                "win_rate",
            ]:
                print(f"  {col}: {best[col]}")


if __name__ == "__main__":
    main()

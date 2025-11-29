#!/usr/bin/env python3
"""
Round-robin tournament for Kaplan variants.
Tests Kaplan and KaplanV2 against all other strategies.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging

import numpy as np
import pandas as pd

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import UniformTokenGenerator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_matchup(
    buyer_type: str,
    seller_type: str,
    num_rounds: int = 20,
    num_periods: int = 5,
    num_tokens: int = 4,
    num_steps: int = 100,
    price_min: int = 0,
    price_max: int = 200,
    seed: int = 42,
) -> dict:
    """Run a single matchup between two strategy types."""

    buyer_profits = []
    seller_profits = []

    for r in range(num_rounds):
        # Fresh token generator per round
        token_gen = UniformTokenGenerator(
            num_tokens, price_min, price_max, seed + r * 1000, num_buyers=2, num_sellers=2
        )

        # Create agents: 2 buyers of buyer_type vs 2 sellers of seller_type
        buyers = []
        sellers = []

        for i in range(2):
            buyer_vals = token_gen.generate_tokens(is_buyer=True)
            agent = create_agent(
                buyer_type,
                player_id=i + 1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=seed + r * 10 + i,
            )
            agent.start_round(buyer_vals)
            buyers.append(agent)

        for i in range(2):
            seller_costs = token_gen.generate_tokens(is_buyer=False)
            agent = create_agent(
                seller_type,
                player_id=i + 3,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_costs,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=seed + r * 10 + i + 2,
            )
            agent.start_round(seller_costs)
            sellers.append(agent)

        round_buyer_profit = 0
        round_seller_profit = 0

        for p in range(num_periods):
            market = Market(
                num_buyers=2,
                num_sellers=2,
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

            for b in buyers:
                round_buyer_profit += b.period_profit
            for s in sellers:
                round_seller_profit += s.period_profit

            for agent in buyers + sellers:
                agent.end_period()

        buyer_profits.append(round_buyer_profit)
        seller_profits.append(round_seller_profit)

    return {
        "buyer_type": buyer_type,
        "seller_type": seller_type,
        "buyer_profit": np.mean(buyer_profits),
        "seller_profit": np.mean(seller_profits),
    }


def run_round_robin(strategies: list[str], num_rounds: int = 20) -> pd.DataFrame:
    """Run round-robin tournament between all strategy pairs."""

    results = []
    total_matchups = len(strategies) * len(strategies)
    count = 0

    print(
        f"Running round-robin tournament: {len(strategies)} strategies, {total_matchups} matchups"
    )
    print("-" * 60)

    for buyer_type in strategies:
        for seller_type in strategies:
            count += 1
            result = run_matchup(buyer_type, seller_type, num_rounds=num_rounds)
            results.append(result)

            if count % 5 == 0:
                print(
                    f"  [{count}/{total_matchups}] {buyer_type} vs {seller_type}: "
                    f"buyer={result['buyer_profit']:.0f}, seller={result['seller_profit']:.0f}"
                )

    return pd.DataFrame(results)


def calculate_rankings(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Calculate overall rankings from round-robin results."""

    rankings = []

    for strategy in strategies:
        # Profit when playing as BUYER
        buyer_rows = df[df["buyer_type"] == strategy]
        avg_buyer_profit = buyer_rows["buyer_profit"].mean()

        # Profit when playing as SELLER
        seller_rows = df[df["seller_type"] == strategy]
        avg_seller_profit = seller_rows["seller_profit"].mean()

        # Total average profit (buyer + seller roles)
        total_profit = avg_buyer_profit + avg_seller_profit

        rankings.append(
            {
                "strategy": strategy,
                "as_buyer": avg_buyer_profit,
                "as_seller": avg_seller_profit,
                "total": total_profit,
            }
        )

    ranking_df = pd.DataFrame(rankings)
    ranking_df = ranking_df.sort_values("total", ascending=False)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    return ranking_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kaplan round-robin tournament")
    parser.add_argument("--num_rounds", type=int, default=20, help="Rounds per matchup")
    parser.add_argument(
        "--strategies",
        type=str,
        default="ZIC,ZIP,Kaplan,KaplanV2,GD,Skeleton",
        help="Comma-separated strategy list",
    )

    args = parser.parse_args()

    strategies = args.strategies.split(",")

    print("=" * 60)
    print("KAPLAN ROUND-ROBIN TOURNAMENT")
    print("=" * 60)
    print(f"Strategies: {strategies}")
    print(f"Rounds per matchup: {args.num_rounds}")
    print()

    # Run tournament
    results_df = run_round_robin(strategies, num_rounds=args.num_rounds)

    # Calculate rankings
    rankings = calculate_rankings(results_df, strategies)

    print("\n" + "=" * 60)
    print("FINAL RANKINGS")
    print("=" * 60)
    print(rankings.to_string(index=False))

    # Show head-to-head for Kaplan variants
    print("\n" + "=" * 60)
    print("KAPLAN HEAD-TO-HEAD")
    print("=" * 60)

    kaplan_variants = [s for s in strategies if "Kaplan" in s]
    for kv in kaplan_variants:
        print(f"\n{kv} as BUYER vs:")
        for opponent in strategies:
            if opponent != kv:
                row = results_df[
                    (results_df["buyer_type"] == kv) & (results_df["seller_type"] == opponent)
                ]
                if not row.empty:
                    diff = row["buyer_profit"].values[0] - row["seller_profit"].values[0]
                    print(f"  {opponent:12s}: {diff:+.0f}")


if __name__ == "__main__":
    main()

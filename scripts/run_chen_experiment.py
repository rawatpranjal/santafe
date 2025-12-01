#!/usr/bin/env python3
"""
Chen et al. (2010) Replication Experiment Runner.

Runs the experimental setup from:
Chen & Tai (2010). "The Agent-Based Double Auction Markets: 15 Years On"

Key features:
- 300 random 4v4 match-ups from strategy pool
- 7,000 trading days per match-up (175,000 time steps each)
- Individual efficiency metric: Actual_Profit / Equilibrium_Profit
- Multi-period learning for adaptive strategies (especially Kaplan)

Usage:
    # Validation run (seconds)
    python scripts/run_chen_experiment.py --matchups 3 --days 70

    # Quick test (minutes)
    python scripts/run_chen_experiment.py --matchups 30 --days 700

    # Full Chen et al. replication (hours)
    python scripts/run_chen_experiment.py --matchups 300 --days 7000
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_equilibrium_profits,
    calculate_max_surplus,
)
from engine.market import Market
from engine.matchup_generator import MatchupGenerator, get_default_strategy_pool
from engine.token_generator import TokenGenerator
from traders.base import Agent


@dataclass
class AgentResult:
    """Results for a single agent across a match-up."""

    agent_id: int
    agent_type: str
    is_buyer: bool
    total_profit: int = 0
    equilibrium_profit: int = 0
    num_trades: int = 0

    @property
    def efficiency_ratio(self) -> float:
        """Individual efficiency = actual / equilibrium profit."""
        if self.equilibrium_profit <= 0:
            return 1.0 if self.total_profit >= 0 else 0.0
        return self.total_profit / self.equilibrium_profit


@dataclass
class MatchupResult:
    """Results for a single match-up."""

    matchup_id: int
    buyer_types: list[str]
    seller_types: list[str]
    num_days: int
    steps_per_day: int
    agent_results: dict[int, AgentResult] = field(default_factory=dict)
    total_trades: int = 0
    market_efficiency: float = 0.0
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "matchup_id": self.matchup_id,
            "buyer_types": self.buyer_types,
            "seller_types": self.seller_types,
            "num_days": self.num_days,
            "steps_per_day": self.steps_per_day,
            "total_trades": self.total_trades,
            "market_efficiency": self.market_efficiency,
            "elapsed_time": self.elapsed_time,
            "agent_results": {
                str(aid): {
                    "agent_id": r.agent_id,
                    "agent_type": r.agent_type,
                    "is_buyer": r.is_buyer,
                    "total_profit": r.total_profit,
                    "equilibrium_profit": r.equilibrium_profit,
                    "num_trades": r.num_trades,
                    "efficiency_ratio": r.efficiency_ratio,
                }
                for aid, r in self.agent_results.items()
            },
        }


def create_agents_for_matchup(
    buyer_types: list[str],
    seller_types: list[str],
    num_tokens: int,
    price_min: int,
    price_max: int,
    num_times: int,
    seed: int,
) -> tuple[list[Agent], list[Agent]]:
    """
    Create agents for a match-up.

    Args:
        buyer_types: List of buyer strategy names
        seller_types: List of seller strategy names
        num_tokens: Tokens per agent
        price_min: Minimum price
        price_max: Maximum price
        num_times: Steps per day (for agent initialization)
        seed: Random seed

    Returns:
        (buyers, sellers) tuple of agent lists
    """
    buyers = []
    sellers = []
    agent_id = 1

    # Create buyers
    for i, agent_type in enumerate(buyer_types):
        agent = create_agent(
            agent_type=agent_type,
            player_id=agent_id,
            is_buyer=True,
            num_tokens=num_tokens,
            valuations=[0] * num_tokens,  # Will be set per day
            seed=seed + agent_id,
            num_times=num_times,
            num_buyers=len(buyer_types),
            num_sellers=len(seller_types),
            price_min=price_min,
            price_max=price_max,
        )
        buyers.append(agent)
        agent_id += 1

    # Create sellers
    for i, agent_type in enumerate(seller_types):
        agent = create_agent(
            agent_type=agent_type,
            player_id=agent_id,
            is_buyer=False,
            num_tokens=num_tokens,
            valuations=[0] * num_tokens,  # Will be set per day
            seed=seed + agent_id,
            num_times=num_times,
            num_buyers=len(buyer_types),
            num_sellers=len(seller_types),
            price_min=price_min,
            price_max=price_max,
        )
        sellers.append(agent)
        agent_id += 1

    return buyers, sellers


def run_single_matchup(
    matchup_id: int,
    buyer_types: list[str],
    seller_types: list[str],
    num_days: int,
    steps_per_day: int,
    num_tokens: int,
    price_min: int,
    price_max: int,
    game_type: int,
    seed: int,
    verbose: bool = False,
) -> MatchupResult:
    """
    Run a single match-up for multiple trading days.

    Args:
        matchup_id: Match-up identifier
        buyer_types: List of buyer strategy names
        seller_types: List of seller strategy names
        num_days: Number of trading days
        steps_per_day: Time steps per day
        num_tokens: Tokens per agent
        price_min: Minimum price
        price_max: Maximum price
        game_type: Token generator game type
        seed: Random seed
        verbose: Print progress

    Returns:
        MatchupResult with all metrics
    """
    start_time = time.time()

    # Create token generator
    token_gen = TokenGenerator(game_type, num_tokens, seed)

    # Create agents (they persist across all days)
    buyers, sellers = create_agents_for_matchup(
        buyer_types=buyer_types,
        seller_types=seller_types,
        num_tokens=num_tokens,
        price_min=price_min,
        price_max=price_max,
        num_times=steps_per_day,
        seed=seed + 1000,
    )

    # Initialize result tracking
    result = MatchupResult(
        matchup_id=matchup_id,
        buyer_types=buyer_types,
        seller_types=seller_types,
        num_days=num_days,
        steps_per_day=steps_per_day,
    )

    # Initialize agent results
    for buyer in buyers:
        result.agent_results[buyer.player_id] = AgentResult(
            agent_id=buyer.player_id,
            agent_type=buyer.__class__.__name__,
            is_buyer=True,
        )
    for seller in sellers:
        result.agent_results[seller.player_id] = AgentResult(
            agent_id=seller.player_id,
            agent_type=seller.__class__.__name__,
            is_buyer=False,
        )

    # Track surplus for efficiency calculation
    total_actual_surplus = 0
    total_max_surplus = 0

    # Run trading days
    for day in range(1, num_days + 1):
        # Generate new tokens for this day
        token_gen.new_round()

        buyer_valuations = []
        for buyer in buyers:
            vals = token_gen.generate_tokens(is_buyer=True)
            buyer_valuations.append(vals)
            buyer.start_round(vals)

        seller_costs = []
        for seller in sellers:
            costs = token_gen.generate_tokens(is_buyer=False)
            seller_costs.append(costs)
            seller.start_round(costs)

        # Calculate equilibrium for this day
        max_surplus = calculate_max_surplus(buyer_valuations, seller_costs)
        total_max_surplus += max_surplus

        # Calculate equilibrium price and profits
        eq_price = calculate_equilibrium_price(buyer_valuations, seller_costs)
        buyer_eq_profits, seller_eq_profits = calculate_equilibrium_profits(
            buyer_valuations, seller_costs, eq_price
        )

        # Accumulate equilibrium profits
        for i, buyer in enumerate(buyers):
            result.agent_results[buyer.player_id].equilibrium_profit += buyer_eq_profits[i + 1]
        for i, seller in enumerate(sellers):
            result.agent_results[seller.player_id].equilibrium_profit += seller_eq_profits[i + 1]

        # Create market for this day
        market = Market(
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            num_times=steps_per_day,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
            seed=seed + day,
        )

        # Notify agents of period start
        for buyer in buyers:
            buyer.start_period(day)
        for seller in sellers:
            seller.start_period(day)

        # Run the trading day
        day_trades = 0
        for _ in range(steps_per_day):
            if not market.run_time_step():
                break

            # Check for trade
            if market.orderbook.trade_price[market.current_time] > 0:
                day_trades += 1
                result.total_trades += 1

        # Collect day profits
        for buyer in buyers:
            result.agent_results[buyer.player_id].total_profit += buyer.period_profit
            result.agent_results[buyer.player_id].num_trades += buyer.num_trades
            total_actual_surplus += buyer.period_profit

        for seller in sellers:
            result.agent_results[seller.player_id].total_profit += seller.period_profit
            result.agent_results[seller.player_id].num_trades += seller.num_trades
            total_actual_surplus += seller.period_profit

        if verbose and day % 100 == 0:
            print(f"  Day {day}/{num_days}: {day_trades} trades")

    # Calculate market efficiency
    if total_max_surplus > 0:
        result.market_efficiency = (total_actual_surplus / total_max_surplus) * 100.0
    else:
        result.market_efficiency = 100.0

    result.elapsed_time = time.time() - start_time

    return result


def calculate_equilibrium_price(
    buyer_valuations: list[list[int]],
    seller_costs: list[list[int]],
) -> int:
    """
    Calculate competitive equilibrium price.

    Uses the intersection of supply and demand curves.
    """
    # Flatten and sort
    all_buyer_vals = sorted([v for vals in buyer_valuations for v in vals], reverse=True)
    all_seller_costs = sorted([c for costs in seller_costs for c in costs])

    # Find intersection
    num_trades = min(len(all_buyer_vals), len(all_seller_costs))

    for i in range(num_trades):
        if all_buyer_vals[i] <= all_seller_costs[i]:
            # Intersection found
            if i == 0:
                return (all_buyer_vals[0] + all_seller_costs[0]) // 2
            else:
                return (all_buyer_vals[i - 1] + all_seller_costs[i - 1]) // 2

    # All trades are profitable, use midpoint of last profitable pair
    if num_trades > 0:
        return (all_buyer_vals[num_trades - 1] + all_seller_costs[num_trades - 1]) // 2

    return 50  # Default midpoint


def run_chen_experiment(
    num_matchups: int = 300,
    num_days: int = 7000,
    steps_per_day: int = 25,
    num_tokens: int = 4,
    num_buyers: int = 4,
    num_sellers: int = 4,
    price_min: int = 0,
    price_max: int = 100,
    game_type: int = 1111,
    strategy_pool: list[str] | None = None,
    matchup_seed: int = 42,
    auction_seed: int = 123,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full Chen et al. (2010) experiment.

    Args:
        num_matchups: Number of random match-ups (default: 300)
        num_days: Trading days per match-up (default: 7000)
        steps_per_day: Time steps per day (default: 25)
        num_tokens: Tokens per agent (default: 4)
        num_buyers: Buyers per match-up (default: 4)
        num_sellers: Sellers per match-up (default: 4)
        price_min: Minimum price (default: 0)
        price_max: Maximum price (default: 100)
        game_type: Token generator type (default: 1111)
        strategy_pool: List of strategies (default: all legacy)
        matchup_seed: Seed for match-up generation (default: 42)
        auction_seed: Seed for auction randomness (default: 123)
        output_path: Path to save results JSON (optional)
        verbose: Print progress (default: True)

    Returns:
        Aggregated results dictionary
    """
    if strategy_pool is None:
        strategy_pool = get_default_strategy_pool()

    # Generate match-ups
    generator = MatchupGenerator(
        strategy_pool=strategy_pool,
        num_matchups=num_matchups,
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        seed=matchup_seed,
    )
    matchups = generator.generate_matchups()

    if verbose:
        print("Chen et al. (2010) Replication Experiment")
        print("=" * 50)
        print(f"Match-ups: {num_matchups}")
        print(f"Days per match-up: {num_days}")
        print(f"Steps per day: {steps_per_day}")
        print(f"Market: {num_buyers}v{num_sellers}")
        print(f"Tokens: {num_tokens}")
        print(f"Strategy pool: {strategy_pool}")
        print(f"Total time steps: {num_matchups * num_days * steps_per_day:,}")
        print("=" * 50)

    # Run all match-ups
    all_results: list[MatchupResult] = []

    start_time = time.time()

    for i, (buyers, sellers) in enumerate(matchups):
        if verbose:
            print(f"\nMatch-up {i+1}/{num_matchups}: {buyers} vs {sellers}")

        result = run_single_matchup(
            matchup_id=i + 1,
            buyer_types=buyers,
            seller_types=sellers,
            num_days=num_days,
            steps_per_day=steps_per_day,
            num_tokens=num_tokens,
            price_min=price_min,
            price_max=price_max,
            game_type=game_type,
            seed=auction_seed + i * 10000,
            verbose=verbose,
        )

        all_results.append(result)

        if verbose:
            print(f"  Completed in {result.elapsed_time:.1f}s")
            print(f"  Market efficiency: {result.market_efficiency:.1f}%")
            print(f"  Total trades: {result.total_trades}")

    total_time = time.time() - start_time

    # Aggregate results by strategy
    strategy_stats: dict[str, dict[str, Any]] = {}

    for result in all_results:
        for agent_result in result.agent_results.values():
            agent_type = agent_result.agent_type
            if agent_type not in strategy_stats:
                strategy_stats[agent_type] = {
                    "total_profit": 0,
                    "equilibrium_profit": 0,
                    "num_trades": 0,
                    "appearances": 0,
                    "efficiency_ratios": [],
                }

            stats = strategy_stats[agent_type]
            stats["total_profit"] += agent_result.total_profit
            stats["equilibrium_profit"] += agent_result.equilibrium_profit
            stats["num_trades"] += agent_result.num_trades
            stats["appearances"] += 1
            stats["efficiency_ratios"].append(agent_result.efficiency_ratio)

    # Calculate summary statistics
    for agent_type, stats in strategy_stats.items():
        ratios = stats["efficiency_ratios"]
        stats["mean_efficiency_ratio"] = np.mean(ratios) if ratios else 0.0
        stats["std_efficiency_ratio"] = np.std(ratios) if ratios else 0.0
        stats["overall_efficiency_ratio"] = (
            stats["total_profit"] / stats["equilibrium_profit"]
            if stats["equilibrium_profit"] > 0
            else 0.0
        )
        # Remove raw ratios for cleaner output
        del stats["efficiency_ratios"]

    # Sort by mean efficiency ratio
    sorted_strategies = sorted(
        strategy_stats.items(), key=lambda x: x[1]["mean_efficiency_ratio"], reverse=True
    )

    # Build final results
    final_results = {
        "config": {
            "num_matchups": num_matchups,
            "num_days": num_days,
            "steps_per_day": steps_per_day,
            "num_tokens": num_tokens,
            "market_size": f"{num_buyers}v{num_sellers}",
            "strategy_pool": strategy_pool,
            "matchup_seed": matchup_seed,
            "auction_seed": auction_seed,
        },
        "summary": {
            "total_elapsed_time": total_time,
            "avg_market_efficiency": np.mean([r.market_efficiency for r in all_results]),
            "total_trades": sum(r.total_trades for r in all_results),
        },
        "strategy_rankings": [
            {"rank": i + 1, "strategy": agent_type, **stats}
            for i, (agent_type, stats) in enumerate(sorted_strategies)
        ],
        "matchup_results": [r.to_dict() for r in all_results],
    }

    # Print summary
    if verbose:
        print(f"\n{'=' * 50}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 50}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg market efficiency: {final_results['summary']['avg_market_efficiency']:.1f}%")
        print("\nStrategy Rankings (by mean efficiency ratio):")
        print("-" * 50)
        for item in final_results["strategy_rankings"]:
            print(
                f"  {item['rank']}. {item['strategy']:12s} "
                f"ratio={item['mean_efficiency_ratio']:.3f} "
                f"(std={item['std_efficiency_ratio']:.3f}, "
                f"n={item['appearances']})"
            )

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return final_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Chen et al. (2010) replication experiment")
    parser.add_argument(
        "--matchups", type=int, default=300, help="Number of random match-ups (default: 300)"
    )
    parser.add_argument(
        "--days", type=int, default=7000, help="Trading days per match-up (default: 7000)"
    )
    parser.add_argument("--steps", type=int, default=25, help="Time steps per day (default: 25)")
    parser.add_argument("--tokens", type=int, default=4, help="Tokens per agent (default: 4)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/chen_2010_<timestamp>.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Strategy pool (default: ZIC Kaplan GD ZIP Ledyard Lin Jacobson Perry)",
    )

    args = parser.parse_args()

    # Default output path with timestamp
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/chen_2010_{timestamp}.json"

    run_chen_experiment(
        num_matchups=args.matchups,
        num_days=args.days,
        steps_per_day=args.steps,
        num_tokens=args.tokens,
        matchup_seed=args.seed,
        auction_seed=args.seed + 1000,
        strategy_pool=args.strategies,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

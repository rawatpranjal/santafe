#!/usr/bin/env python3
"""
Evolutionary Tournament for Double Auction Trading Strategies.

Tests the hypothesis: "Snipers (Kaplan) need noise traders (ZIC) to survive."

Runs co-evolutionary dynamics on 9 legacy strategies from Part 2 of the paper:
ZIC, Skeleton, ZIP, Kaplan, Ringuette, GD, Ledyard, BGAN, Staecker

Usage:
    # Quick test (5 generations)
    python scripts/run_evolutionary_tournament.py --generations 5 --matchups 10

    # Full run (100 generations, 10 seeds)
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        python scripts/run_evolutionary_tournament.py --generations 100 --matchups 30 --seed $seed
    done
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_chen_experiment import run_single_matchup

# Strategy pool for Part 2: Santa Fe 1991 traders ONLY
# (No ZIP - Cliff 1997, No GD - Gjerstad-Dickhaut 1998)
# - Baseline: ZIC
# - Snipers: Skeleton, Kaplan, Ringuette
# - Fixed-margin: Gamer
# - Adaptive: Perry, Lin, Breton
# - Belief-based: BGAN
# - Theory-based: Ledyard (EL), Staecker, Jacobson
STRATEGY_POOL = [
    "ZIC",  # Baseline (Gode & Sunder)
    "ZIP",  # Cliff 1997 (added per user request)
    "Skeleton",  # Default code
    "Kaplan",  # Winner
    "Ringuette",  # 2nd place
    "Gamer",  # Fixed margin
    "Perry",  # Efficiency-based learning
    "Lin",  # Statistical prediction
    "Breton",  # Stochastic adaptive
    "BGAN",  # Bayesian game against nature
    "Ledyard",  # Easley-Ledyard (EL)
    "Staecker",  # Predictive (exponential smoothing)
    "Jacobson",  # Equilibrium estimation
]


@dataclass
class PopulationAgent:
    """An individual in the evolving population."""

    agent_id: int
    strategy: str
    total_profit: float = 0.0
    num_matchups: int = 0

    @property
    def avg_profit(self) -> float:
        """Average profit per matchup."""
        if self.num_matchups == 0:
            return 0.0
        return self.total_profit / self.num_matchups

    def reset_profit(self) -> None:
        """Reset profit tracking for new generation."""
        self.total_profit = 0.0
        self.num_matchups = 0


@dataclass
class GenerationStats:
    """Statistics for one generation."""

    generation: int
    strategy_counts: dict[str, int]
    mean_efficiency: float
    total_trades: int
    mean_profit_by_strategy: dict[str, float]
    std_profit_by_strategy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "generation": self.generation,
            "strategy_counts": self.strategy_counts,
            "mean_efficiency": self.mean_efficiency,
            "total_trades": self.total_trades,
            "mean_profit_by_strategy": self.mean_profit_by_strategy,
            "std_profit_by_strategy": self.std_profit_by_strategy,
        }


def initialize_population(
    population_size: int,
    strategies: list[str],
    seed: int,
) -> list[PopulationAgent]:
    """
    Initialize population with uniform distribution across strategies.

    Args:
        population_size: Total number of agents
        strategies: List of strategy types
        seed: Random seed

    Returns:
        List of PopulationAgent instances
    """
    rng = random.Random(seed)

    # Calculate agents per strategy (uniform distribution)
    agents_per_strategy = population_size // len(strategies)
    remainder = population_size % len(strategies)

    population = []
    agent_id = 0

    for i, strategy in enumerate(strategies):
        # Add one extra agent to first 'remainder' strategies
        count = agents_per_strategy + (1 if i < remainder else 0)
        for _ in range(count):
            population.append(PopulationAgent(agent_id=agent_id, strategy=strategy))
            agent_id += 1

    # Shuffle to randomize initial positions
    rng.shuffle(population)
    return population


def get_strategy_distribution(population: list[PopulationAgent]) -> dict[str, int]:
    """Count agents by strategy type."""
    counts: dict[str, int] = {}
    for agent in population:
        counts[agent.strategy] = counts.get(agent.strategy, 0) + 1
    return counts


def run_generation(
    population: list[PopulationAgent],
    generation: int,
    matchups_per_gen: int,
    num_periods: int,
    steps_per_period: int,
    num_tokens: int,
    price_min: int,
    price_max: int,
    game_type: int,
    seed: int,
    verbose: bool = False,
) -> GenerationStats:
    """
    Run one generation of evolution.

    Args:
        population: List of agents
        generation: Generation number
        matchups_per_gen: Number of matchups per generation
        num_periods: Trading periods per matchup
        steps_per_period: Time steps per period
        num_tokens: Tokens per agent
        price_min: Minimum price
        price_max: Maximum price
        game_type: Token generator type
        seed: Random seed
        verbose: Print progress

    Returns:
        GenerationStats for this generation
    """
    rng = random.Random(seed + generation * 1000)

    # Reset all agent profits
    for agent in population:
        agent.reset_profit()

    # Track generation metrics
    total_efficiency = 0.0
    total_trades = 0
    matchup_count = 0

    # Run matchups
    for matchup_idx in range(matchups_per_gen):
        # Sample 8 agents from population
        if len(population) < 8:
            raise ValueError(f"Population too small: {len(population)} < 8")

        participants = rng.sample(population, 8)

        # Randomly assign roles (4 buyers, 4 sellers)
        rng.shuffle(participants)
        buyers = participants[:4]
        sellers = participants[4:]

        buyer_types = [a.strategy for a in buyers]
        seller_types = [a.strategy for a in sellers]

        # Run the matchup using existing infrastructure
        result = run_single_matchup(
            matchup_id=matchup_idx,
            buyer_types=buyer_types,
            seller_types=seller_types,
            num_days=num_periods,
            steps_per_day=steps_per_period,
            num_tokens=num_tokens,
            price_min=price_min,
            price_max=price_max,
            game_type=game_type,
            seed=seed + generation * 10000 + matchup_idx * 100,
            verbose=False,
        )

        # Distribute profits back to population agents
        for i, buyer in enumerate(buyers):
            agent_result = result.agent_results[i + 1]
            buyer.total_profit += agent_result.total_profit
            buyer.num_matchups += 1

        for i, seller in enumerate(sellers):
            agent_result = result.agent_results[i + 5]
            seller.total_profit += agent_result.total_profit
            seller.num_matchups += 1

        total_efficiency += result.market_efficiency
        total_trades += result.total_trades
        matchup_count += 1

        if verbose and (matchup_idx + 1) % 10 == 0:
            print(f"  Matchup {matchup_idx + 1}/{matchups_per_gen}")

    # Calculate strategy-level statistics
    strategy_profits: dict[str, list[float]] = {s: [] for s in STRATEGY_POOL}
    for agent in population:
        if agent.num_matchups > 0:
            strategy_profits[agent.strategy].append(agent.avg_profit)

    mean_profit_by_strategy = {}
    std_profit_by_strategy = {}
    for strategy, profits in strategy_profits.items():
        if profits:
            mean_profit_by_strategy[strategy] = float(np.mean(profits))
            std_profit_by_strategy[strategy] = float(np.std(profits))
        else:
            mean_profit_by_strategy[strategy] = 0.0
            std_profit_by_strategy[strategy] = 0.0

    return GenerationStats(
        generation=generation,
        strategy_counts=get_strategy_distribution(population),
        mean_efficiency=total_efficiency / matchup_count if matchup_count > 0 else 0.0,
        total_trades=total_trades,
        mean_profit_by_strategy=mean_profit_by_strategy,
        std_profit_by_strategy=std_profit_by_strategy,
    )


def apply_selection(
    population: list[PopulationAgent],
    elimination_rate: float,
    immigration_count: int,
    seed: int,
) -> list[PopulationAgent]:
    """
    Apply selection: eliminate bottom performers, clone top performers, add immigrants.

    Args:
        population: Current population
        elimination_rate: Fraction to eliminate (e.g., 0.25 = bottom 25%)
        immigration_count: Number of random immigrants to add each generation
        seed: Random seed for tie-breaking

    Returns:
        New population after selection
    """
    rng = random.Random(seed)

    # Sort by average profit (ascending)
    sorted_pop = sorted(population, key=lambda a: a.avg_profit)

    num_eliminate = int(len(population) * elimination_rate)

    # Bottom N are eliminated
    eliminated = sorted_pop[:num_eliminate]

    # Top N will reproduce
    top_performers = sorted_pop[-num_eliminate:] if num_eliminate > 0 else []

    # Middle survives unchanged
    survivors = sorted_pop[num_eliminate:-num_eliminate] if num_eliminate > 0 else sorted_pop

    # Create replacements: some from cloning top performers, some from immigration
    replacements = []
    next_id = max(a.agent_id for a in population) + 1

    # Immigration: add random strategy agents
    num_immigrants = min(immigration_count, num_eliminate)
    for i in range(num_immigrants):
        new_strategy = rng.choice(STRATEGY_POOL)
        new_agent = PopulationAgent(
            agent_id=next_id,
            strategy=new_strategy,
        )
        replacements.append(new_agent)
        next_id += 1

    # Clone top performers for remaining slots
    num_clones = num_eliminate - num_immigrants
    for i in range(num_clones):
        if top_performers:
            parent = rng.choice(top_performers)
            old_agent = (
                eliminated[num_immigrants + i]
                if (num_immigrants + i) < len(eliminated)
                else eliminated[i % len(eliminated)]
            )
            new_agent = PopulationAgent(
                agent_id=old_agent.agent_id,
                strategy=parent.strategy,
            )
            replacements.append(new_agent)

    # Combine survivors and replacements
    new_population = list(survivors) + list(top_performers) + replacements
    return new_population


def run_evolutionary_tournament(
    population_size: int = 32,
    num_generations: int = 50,
    matchups_per_gen: int = 30,
    num_periods: int = 10,
    steps_per_period: int = 100,
    num_tokens: int = 4,
    price_min: int = 1,
    price_max: int = 1000,
    game_type: int = 6453,
    elimination_rate: float = 0.25,
    immigration_count: int = 0,
    seed: int = 42,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full evolutionary tournament.

    Args:
        population_size: Number of agents in population
        num_generations: Number of generations to run
        matchups_per_gen: Matchups per generation
        num_periods: Trading periods per matchup
        steps_per_period: Time steps per period
        num_tokens: Tokens per agent
        price_min: Minimum price
        price_max: Maximum price
        game_type: Token generator type
        elimination_rate: Fraction to eliminate each generation
        immigration_count: Number of random immigrants per generation
        seed: Random seed
        output_path: Path to save results JSON
        verbose: Print progress

    Returns:
        Results dictionary
    """
    if verbose:
        print("Evolutionary Tournament")
        print("=" * 50)
        print(f"Population: {population_size}")
        print(f"Generations: {num_generations}")
        print(f"Matchups/gen: {matchups_per_gen}")
        print(f"Periods/matchup: {num_periods}")
        print(f"Steps/period: {steps_per_period}")
        print(f"Strategies: {STRATEGY_POOL}")
        print(f"Elimination rate: {elimination_rate:.0%}")
        print(f"Immigration: {immigration_count}/gen")
        print("=" * 50)

    # Initialize population
    population = initialize_population(population_size, STRATEGY_POOL, seed)

    history: list[GenerationStats] = []
    extinction_events: list[dict[str, Any]] = []
    active_strategies = set(STRATEGY_POOL)

    start_time = time.time()

    for gen in range(num_generations):
        if verbose:
            print(f"\nGeneration {gen}/{num_generations}")
            dist = get_strategy_distribution(population)
            print(f"  Distribution: {dist}")

        # Run generation
        stats = run_generation(
            population=population,
            generation=gen,
            matchups_per_gen=matchups_per_gen,
            num_periods=num_periods,
            steps_per_period=steps_per_period,
            num_tokens=num_tokens,
            price_min=price_min,
            price_max=price_max,
            game_type=game_type,
            seed=seed,
            verbose=verbose,
        )
        history.append(stats)

        if verbose:
            print(f"  Efficiency: {stats.mean_efficiency:.1f}%")
            print(f"  Trades: {stats.total_trades}")

        # Check for extinctions
        for strategy in list(active_strategies):
            if stats.strategy_counts.get(strategy, 0) == 0:
                extinction_events.append({"strategy": strategy, "generation": gen})
                active_strategies.remove(strategy)
                if verbose:
                    print(f"  EXTINCTION: {strategy}")

        # Apply selection (except last generation)
        if gen < num_generations - 1:
            population = apply_selection(
                population, elimination_rate, immigration_count, seed + gen
            )

    elapsed_time = time.time() - start_time

    # Final statistics
    final_dist = get_strategy_distribution(population)

    if verbose:
        print("\n" + "=" * 50)
        print("EVOLUTION COMPLETE")
        print("=" * 50)
        print(f"Elapsed time: {elapsed_time:.1f}s")
        print(f"Final distribution: {final_dist}")
        print(f"Extinctions: {[e['strategy'] for e in extinction_events]}")

    # Build results
    results = {
        "config": {
            "population_size": population_size,
            "num_generations": num_generations,
            "matchups_per_gen": matchups_per_gen,
            "num_periods": num_periods,
            "steps_per_period": steps_per_period,
            "num_tokens": num_tokens,
            "price_min": price_min,
            "price_max": price_max,
            "game_type": game_type,
            "elimination_rate": elimination_rate,
            "immigration_count": immigration_count,
            "strategies": STRATEGY_POOL,
            "seed": seed,
        },
        "summary": {
            "elapsed_time": elapsed_time,
            "final_distribution": final_dist,
            "extinction_events": extinction_events,
            "surviving_strategies": list(active_strategies),
        },
        "generations": [s.to_dict() for s in history],
    }

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evolutionary tournament for trading strategies"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=32,
        help="Population size (default: 32)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (default: 50)",
    )
    parser.add_argument(
        "--matchups",
        type=int,
        default=30,
        help="Matchups per generation (default: 30)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=10,
        help="Trading periods per matchup (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Time steps per period (default: 100)",
    )
    parser.add_argument(
        "--elimination",
        type=float,
        default=0.25,
        help="Elimination rate (default: 0.25)",
    )
    parser.add_argument(
        "--immigration",
        type=int,
        default=0,
        help="Number of random immigrants per generation (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/evolution_<timestamp>.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Default output path with timestamp
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/evolution_{timestamp}.json"

    run_evolutionary_tournament(
        population_size=args.population,
        num_generations=args.generations,
        matchups_per_gen=args.matchups,
        num_periods=args.periods,
        steps_per_period=args.steps,
        elimination_rate=args.elimination,
        immigration_count=args.immigration,
        seed=args.seed,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

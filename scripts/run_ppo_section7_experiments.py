#!/usr/bin/env python3
"""
PPO Section 7 Experiments - Matches Part 2 Format.

This script runs control and pairwise experiments for PPO to match
the experimental framework used in Section 6 (Part 2).

Control experiments: 1 PPO buyer + 3 ZIC buyers + 4 ZIC sellers
Pairwise experiments: 4 PPO buyers + 4 opponent buyers vs 4 opponent sellers

Outputs JSON results for table generation.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime

import numpy as np

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_allocative_efficiency,
    calculate_max_surplus,
    get_transaction_prices,
)
from engine.market import Market
from engine.token_generator import TokenGenerator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Environment configurations matching Part 2
ENVIRONMENTS = {
    "BASE": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_steps": 100,
    },
    "BBBS": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 6,
        "num_sellers": 2,
        "num_steps": 100,
    },
    "BSSS": {
        "gametype": 6453,
        "num_tokens": 4,
        "num_buyers": 2,
        "num_sellers": 6,
        "num_steps": 100,
    },
    "EQL": {"gametype": 1111, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "RAN": {"gametype": 9999, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "PER": {"gametype": 6453, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "SHRT": {"gametype": 6453, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 20},
    "TOK": {"gametype": 6453, "num_tokens": 8, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "SML": {"gametype": 1111, "num_tokens": 2, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
    "LAD": {"gametype": 9731, "num_tokens": 4, "num_buyers": 4, "num_sellers": 4, "num_steps": 100},
}

# Pairwise opponents
PAIRWISE_OPPONENTS = ["ZIC", "ZIP", "Skeleton", "Kaplan"]


def run_control_experiment(
    env_name: str,
    env_config: dict,
    model_path: str,
    num_rounds: int = 50,
    num_periods: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run control experiment: 1 PPO buyer + 3 ZIC buyers + 4 ZIC sellers.

    Returns efficiency, volatility, and profit metrics.
    """
    gametype = env_config["gametype"]
    num_tokens = env_config["num_tokens"]
    num_buyers = env_config["num_buyers"]
    num_sellers = env_config["num_sellers"]
    num_steps = env_config["num_steps"]

    np.random.seed(seed)

    efficiencies = []
    volatilities = []
    ppo_profits = []
    zic_profits = []

    for round_num in range(num_rounds):
        round_seed = seed + round_num * 1000

        # Generate tokens
        token_gen = TokenGenerator(gametype, num_tokens, round_seed)
        token_gen.new_round()

        # Store tokens separately for efficiency calculation
        buyer_tokens_list = []
        seller_tokens_list = []

        # Create buyers: 1 PPO + (num_buyers-1) ZIC
        buyers = []

        # PPO buyer
        ppo_tokens = token_gen.generate_tokens(True)
        buyer_tokens_list.append(list(ppo_tokens))
        ppo_agent = create_agent(
            "PPO",
            1,
            True,
            num_tokens,
            ppo_tokens,
            seed=round_seed,
            num_times=num_steps,
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=0,
            price_max=1000,
            model_path=model_path,
        )
        buyers.append(ppo_agent)

        # ZIC buyers
        for i in range(num_buyers - 1):
            zic_tokens = token_gen.generate_tokens(True)
            buyer_tokens_list.append(list(zic_tokens))
            zic_agent = create_agent(
                "ZIC",
                i + 2,
                True,
                num_tokens,
                zic_tokens,
                seed=round_seed + i + 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            buyers.append(zic_agent)

        # ZIC sellers
        sellers = []
        for i in range(num_sellers):
            zic_tokens = token_gen.generate_tokens(False)
            seller_tokens_list.append(list(zic_tokens))
            zic_agent = create_agent(
                "ZIC",
                num_buyers + i + 1,
                False,
                num_tokens,
                zic_tokens,
                seed=round_seed + i + 200,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(zic_agent)

        all_agents = buyers + sellers

        round_efficiency = []
        round_prices = []
        round_ppo_profit = 0
        round_zic_buyer_profit = 0

        for period in range(1, num_periods + 1):
            # Reset period state
            for agent in all_agents:
                agent.start_period(period)

            # Create market
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=0,
                price_max=1000,
                buyers=buyers,
                sellers=sellers,
                seed=round_seed + period * 10000,
            )

            # Run market
            for step in range(num_steps):
                market.run_time_step()

            # Calculate efficiency using stored token lists
            actual_surplus = sum(a.period_profit for a in all_agents)
            max_surplus = calculate_max_surplus(buyer_tokens_list, seller_tokens_list)
            eff = (
                calculate_allocative_efficiency(actual_surplus, max_surplus) / 100.0
            )  # Convert to 0-1
            round_efficiency.append(eff)

            # Collect prices for volatility using get_transaction_prices
            period_prices = get_transaction_prices(market.orderbook, num_steps)
            round_prices.extend(period_prices)

            # Collect profits
            round_ppo_profit += buyers[0].period_profit
            for b in buyers[1:]:
                round_zic_buyer_profit += b.period_profit

        efficiencies.append(np.mean(round_efficiency))

        # Calculate volatility (std of prices / mean of prices)
        if len(round_prices) > 1:
            volatility = np.std(round_prices) / np.mean(round_prices) * 100
        else:
            volatility = 0.0
        volatilities.append(volatility)

        ppo_profits.append(round_ppo_profit)
        zic_profits.append(round_zic_buyer_profit / (num_buyers - 1) if num_buyers > 1 else 0)

    # Calculate invasibility ratio
    mean_ppo = np.mean(ppo_profits)
    mean_zic = np.mean(zic_profits)
    invasibility = mean_ppo / mean_zic if mean_zic > 0 else float("inf")

    return {
        "efficiency_mean": float(np.mean(efficiencies) * 100),
        "efficiency_std": float(np.std(efficiencies) * 100),
        "volatility_mean": float(np.mean(volatilities)),
        "volatility_std": float(np.std(volatilities)),
        "ppo_profit_mean": float(mean_ppo),
        "ppo_profit_std": float(np.std(ppo_profits)),
        "zic_profit_mean": float(mean_zic),
        "zic_profit_std": float(np.std(zic_profits)),
        "invasibility": float(invasibility),
    }


def run_pairwise_experiment(
    opponent: str,
    model_path: str,
    num_rounds: int = 50,
    num_periods: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run pairwise experiment: 4 PPO buyers + 4 opponent (2 buyer, 2 seller).

    Actually: 2 PPO buyers, 2 opponent buyers, 2 PPO-equivalent sellers (ZIC), 2 opponent sellers
    Since PPO is buyer-only, we use: 2 PPO buyers, 2 opponent buyers vs 4 opponent sellers
    """
    env_config = ENVIRONMENTS["BASE"]
    gametype = env_config["gametype"]
    num_tokens = env_config["num_tokens"]
    num_buyers = 4
    num_sellers = 4
    num_steps = env_config["num_steps"]

    np.random.seed(seed)

    efficiencies = []
    ppo_profits = []
    opponent_profits = []

    for round_num in range(num_rounds):
        round_seed = seed + round_num * 1000

        token_gen = TokenGenerator(gametype, num_tokens, round_seed)
        token_gen.new_round()

        # Store tokens separately for efficiency calculation
        buyer_tokens_list = []
        seller_tokens_list = []

        # Buyers: 2 PPO + 2 opponent
        buyers = []

        for i in range(2):
            ppo_tokens = token_gen.generate_tokens(True)
            buyer_tokens_list.append(list(ppo_tokens))
            ppo_agent = create_agent(
                "PPO",
                i + 1,
                True,
                num_tokens,
                ppo_tokens,
                seed=round_seed + i,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
                model_path=model_path,
            )
            buyers.append(ppo_agent)

        for i in range(2):
            opp_tokens = token_gen.generate_tokens(True)
            buyer_tokens_list.append(list(opp_tokens))
            opp_agent = create_agent(
                opponent,
                i + 3,
                True,
                num_tokens,
                opp_tokens,
                seed=round_seed + i + 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            buyers.append(opp_agent)

        # Sellers: 4 opponent
        sellers = []
        for i in range(num_sellers):
            opp_tokens = token_gen.generate_tokens(False)
            seller_tokens_list.append(list(opp_tokens))
            opp_agent = create_agent(
                opponent,
                num_buyers + i + 1,
                False,
                num_tokens,
                opp_tokens,
                seed=round_seed + i + 200,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(opp_agent)

        all_agents = buyers + sellers

        round_efficiency = []
        round_ppo_profit = 0
        round_opp_profit = 0

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
                seed=round_seed + period * 10000,
            )

            for step in range(num_steps):
                market.run_time_step()

            # Calculate efficiency using stored token lists
            actual_surplus = sum(a.period_profit for a in all_agents)
            max_surplus = calculate_max_surplus(buyer_tokens_list, seller_tokens_list)
            eff = (
                calculate_allocative_efficiency(actual_surplus, max_surplus) / 100.0
            )  # Convert to 0-1
            round_efficiency.append(eff)

            # PPO profits (first 2 buyers)
            round_ppo_profit += buyers[0].period_profit + buyers[1].period_profit
            # Opponent profits (buyers 3-4 + all sellers)
            round_opp_profit += buyers[2].period_profit + buyers[3].period_profit
            for s in sellers:
                round_opp_profit += s.period_profit

        efficiencies.append(np.mean(round_efficiency))
        ppo_profits.append(round_ppo_profit / 2)  # Per-agent average
        opponent_profits.append(round_opp_profit / 6)  # Per-agent average (2 buyers + 4 sellers)

    return {
        "opponent": opponent,
        "efficiency_mean": float(np.mean(efficiencies) * 100),
        "efficiency_std": float(np.std(efficiencies) * 100),
        "ppo_profit_mean": float(np.mean(ppo_profits)),
        "ppo_profit_std": float(np.std(ppo_profits)),
        "opponent_profit_mean": float(np.mean(opponent_profits)),
        "opponent_profit_std": float(np.std(opponent_profits)),
        "profit_ratio": (
            float(np.mean(ppo_profits) / np.mean(opponent_profits))
            if np.mean(opponent_profits) > 0
            else float("inf")
        ),
    }


def run_control_section(
    model_path: str, num_seeds: int = 5, output_dir: str = "results/ppo_section7"
):
    """Run control experiments across all environments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for env_name, env_config in ENVIRONMENTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Control Experiment: PPO vs 7 ZIC in {env_name}")
        logger.info(f"{'='*60}")

        env_results = []
        for seed_idx in range(num_seeds):
            logger.info(f"  Seed {seed_idx + 1}/{num_seeds}...")
            result = run_control_experiment(
                env_name=env_name,
                env_config=env_config,
                model_path=model_path,
                num_rounds=50,
                num_periods=10,
                seed=42 + seed_idx * 10000,
            )
            env_results.append(result)

        # Aggregate across seeds
        results[env_name] = {
            "efficiency_mean": float(np.mean([r["efficiency_mean"] for r in env_results])),
            "efficiency_std": float(np.mean([r["efficiency_std"] for r in env_results])),
            "volatility_mean": float(np.mean([r["volatility_mean"] for r in env_results])),
            "volatility_std": float(np.mean([r["volatility_std"] for r in env_results])),
            "ppo_profit_mean": float(np.mean([r["ppo_profit_mean"] for r in env_results])),
            "ppo_profit_std": float(np.std([r["ppo_profit_mean"] for r in env_results])),
            "zic_profit_mean": float(np.mean([r["zic_profit_mean"] for r in env_results])),
            "zic_profit_std": float(np.std([r["zic_profit_mean"] for r in env_results])),
            "invasibility": float(np.mean([r["invasibility"] for r in env_results])),
        }

        logger.info(
            f"  Efficiency: {results[env_name]['efficiency_mean']:.1f}% +/- {results[env_name]['efficiency_std']:.1f}%"
        )
        logger.info(f"  Volatility: {results[env_name]['volatility_mean']:.1f}%")
        logger.info(f"  Invasibility: {results[env_name]['invasibility']:.2f}x")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"control_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nControl results saved to: {results_file}")

    return results


def run_pairwise_section(
    model_path: str, num_seeds: int = 5, output_dir: str = "results/ppo_section7"
):
    """Run pairwise experiments against each opponent."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for opponent in PAIRWISE_OPPONENTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pairwise Experiment: PPO vs {opponent}")
        logger.info(f"{'='*60}")

        opp_results = []
        for seed_idx in range(num_seeds):
            logger.info(f"  Seed {seed_idx + 1}/{num_seeds}...")
            result = run_pairwise_experiment(
                opponent=opponent,
                model_path=model_path,
                num_rounds=50,
                num_periods=10,
                seed=42 + seed_idx * 10000,
            )
            opp_results.append(result)

        # Aggregate across seeds
        results[opponent] = {
            "efficiency_mean": float(np.mean([r["efficiency_mean"] for r in opp_results])),
            "efficiency_std": float(np.std([r["efficiency_mean"] for r in opp_results])),
            "ppo_profit_mean": float(np.mean([r["ppo_profit_mean"] for r in opp_results])),
            "ppo_profit_std": float(np.std([r["ppo_profit_mean"] for r in opp_results])),
            "opponent_profit_mean": float(
                np.mean([r["opponent_profit_mean"] for r in opp_results])
            ),
            "opponent_profit_std": float(np.std([r["opponent_profit_mean"] for r in opp_results])),
            "profit_ratio": float(np.mean([r["profit_ratio"] for r in opp_results])),
        }

        logger.info(f"  Efficiency: {results[opponent]['efficiency_mean']:.1f}%")
        logger.info(f"  PPO Profit: {results[opponent]['ppo_profit_mean']:.1f}")
        logger.info(f"  {opponent} Profit: {results[opponent]['opponent_profit_mean']:.1f}")
        logger.info(f"  Ratio: {results[opponent]['profit_ratio']:.2f}x")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"pairwise_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nPairwise results saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO Section 7 experiments")
    parser.add_argument(
        "--section",
        type=str,
        choices=["control", "pairwise", "all"],
        default="all",
        help="Which experiments to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_v10_10M/ppo_double_auction_8000000_steps.zip",
        help="Path to PPO buyer model",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ppo_section7",
        help="Output directory for results",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PPO SECTION 7 EXPERIMENTS")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Section: {args.section}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info("=" * 60)

    if args.section in ["control", "all"]:
        run_control_section(args.model, args.seeds, args.output)

    if args.section in ["pairwise", "all"]:
        run_pairwise_section(args.model, args.seeds, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("=" * 60)

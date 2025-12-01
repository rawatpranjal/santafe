#!/usr/bin/env python3
"""
Run the full Santa Fe Tournament replication across all 10 environments.
This script executes all tournament configurations and collects results.

Usage:
    python scripts/run_full_tournament.py [--phase PHASE]

    Phases:
        1: Baseline pure market experiments
        2: Pairwise tournaments
        3: Full 10-environment tournament
        all: Run all phases sequentially
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from engine.market import Market

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FullTournamentRunner:
    """Runs the complete Santa Fe tournament replication."""

    def __init__(self, output_dir: str = None):
        """Initialize the tournament runner."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/tournament_full_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define all trader types (all available legacy traders)
        self.trader_types = [
            # Zero-Intelligence family
            "ZI",
            "ZIC",
            "ZIP",
            "ZI2",
            # Santa Fe Tournament winners/notables
            "Kaplan",
            "Ringuette",
            "Skeleton",
            # Adaptive traders
            "GD",
            "Ledyard",
            # Other legacy traders
            "Lin",
            "Perry",
            "Jacobson",
            "Markup",
            "TruthTeller",
            "Gamer",
            "Breton",
            "Gradual",
            "ReservationPrice",
            "HistogramLearner",
            "RuleTrader",
        ]

        # Define all environments from the 1994 tournament
        self.environments = {
            "BASE": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100},
            "BBBS": {"buyers": 2, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100},
            "BSSS": {"buyers": 4, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100},
            "EQL": {
                "buyers": 4,
                "sellers": 4,
                "tokens": 4,
                "periods": 10,
                "steps": 100,
                "symmetric": True,
            },
            "RAN": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 100},
            "PER": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 1, "steps": 100},
            "SHRT": {"buyers": 4, "sellers": 4, "tokens": 4, "periods": 10, "steps": 20},
            "TOK": {"buyers": 4, "sellers": 4, "tokens": 1, "periods": 10, "steps": 100},
            "SML": {"buyers": 2, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100},
            "LAD": {"buyers": 6, "sellers": 2, "tokens": 4, "periods": 10, "steps": 100},
        }

    def run_phase_1_baseline(self):
        """Phase 1: Run baseline pure market experiments."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Baseline Pure Market Experiments")
        logger.info("=" * 60)

        results = []

        for trader_type in self.trader_types:
            logger.info(f"\nTesting pure {trader_type} market...")

            # Run pure market with only one trader type
            efficiency_values = []
            for round_num in range(10):  # 10 rounds for statistical significance
                market = Market(
                    num_buyers=4,
                    num_sellers=4,
                    num_tokens=4,
                    num_periods=10,
                    num_steps=100,
                    min_price=1,
                    max_price=1000,
                )

                # Create homogeneous market
                for i in range(4):
                    market.add_buyer(trader_type, f"B{i}")
                    market.add_seller(trader_type, f"S{i}")

                # Run market
                market.run()

                # Calculate efficiency
                efficiency = market.get_efficiency()
                efficiency_values.append(efficiency)

            mean_eff = np.mean(efficiency_values)
            std_eff = np.std(efficiency_values)

            logger.info(f"{trader_type}: {mean_eff:.1f}% ± {std_eff:.1f}%")

            results.append(
                {
                    "trader": trader_type,
                    "efficiency_mean": mean_eff,
                    "efficiency_std": std_eff,
                    "num_rounds": 10,
                }
            )

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "phase1_baseline_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nPhase 1 results saved to {output_file}")

        return results

    def run_phase_2_pairwise(self):
        """Phase 2: Run pairwise tournaments."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Pairwise Tournaments")
        logger.info("=" * 60)

        results = []

        # Test all pairwise combinations
        for i, trader1 in enumerate(self.trader_types):
            for trader2 in self.trader_types[i + 1 :]:
                logger.info(f"\nTesting {trader1} vs {trader2}...")

                efficiency_values = []
                profit_shares = {trader1: [], trader2: []}

                for round_num in range(10):
                    market = Market(
                        num_buyers=4,
                        num_sellers=4,
                        num_tokens=4,
                        num_periods=10,
                        num_steps=100,
                        min_price=1,
                        max_price=1000,
                    )

                    # Create mixed market
                    for i in range(2):
                        market.add_buyer(trader1, f"B{i}")
                        market.add_seller(trader1, f"S{i}")
                    for i in range(2, 4):
                        market.add_buyer(trader2, f"B{i}")
                        market.add_seller(trader2, f"S{i}")

                    # Run market
                    market.run()

                    # Calculate metrics
                    efficiency = market.get_efficiency()
                    efficiency_values.append(efficiency)

                    # Calculate profit shares
                    trader1_profit = market.get_trader_profit(trader1)
                    trader2_profit = market.get_trader_profit(trader2)
                    total_profit = trader1_profit + trader2_profit

                    if total_profit > 0:
                        profit_shares[trader1].append(trader1_profit / total_profit * 100)
                        profit_shares[trader2].append(trader2_profit / total_profit * 100)

                mean_eff = np.mean(efficiency_values)
                std_eff = np.std(efficiency_values)

                winner = (
                    trader1
                    if np.mean(profit_shares[trader1]) > np.mean(profit_shares[trader2])
                    else trader2
                )
                winner_share = max(np.mean(profit_shares[trader1]), np.mean(profit_shares[trader2]))

                logger.info(f"Efficiency: {mean_eff:.1f}% ± {std_eff:.1f}%")
                logger.info(f"Winner: {winner} ({winner_share:.1f}% profit share)")

                results.append(
                    {
                        "matchup": f"{trader1}_vs_{trader2}",
                        "efficiency_mean": mean_eff,
                        "efficiency_std": std_eff,
                        "winner": winner,
                        "winner_share": winner_share,
                    }
                )

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "phase2_pairwise_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nPhase 2 results saved to {output_file}")

        return results

    def run_phase_3_full_tournament(self):
        """Phase 3: Run full 10-environment tournament."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Full Tournament (10 Environments)")
        logger.info("=" * 60)

        all_results = []

        for env_name, env_config in self.environments.items():
            logger.info(f"\n--- Environment: {env_name} ---")
            logger.info(f"Config: {env_config}")

            efficiency_values = []
            trader_profits = {trader: [] for trader in self.trader_types}

            for round_num in range(100):  # 100 rounds per environment
                market = Market(
                    num_buyers=env_config["buyers"],
                    num_sellers=env_config["sellers"],
                    num_tokens=env_config.get("tokens", 4),
                    num_periods=env_config.get("periods", 10),
                    num_steps=env_config.get("steps", 100),
                    min_price=1,
                    max_price=1000,
                )

                # Create mixed market with all trader types
                traders_per_side = min(env_config["buyers"], env_config["sellers"])
                for i in range(env_config["buyers"]):
                    trader_type = self.trader_types[i % len(self.trader_types)]
                    market.add_buyer(trader_type, f"B{i}")

                for i in range(env_config["sellers"]):
                    trader_type = self.trader_types[i % len(self.trader_types)]
                    market.add_seller(trader_type, f"S{i}")

                # Run market
                market.run()

                # Calculate efficiency
                efficiency = market.get_efficiency()
                efficiency_values.append(efficiency)

                # Track trader profits
                for trader in self.trader_types:
                    profit = market.get_trader_profit(trader)
                    trader_profits[trader].append(profit)

            mean_eff = np.mean(efficiency_values)
            std_eff = np.std(efficiency_values)

            # Find top trader
            mean_profits = {trader: np.mean(profits) for trader, profits in trader_profits.items()}
            top_trader = max(mean_profits, key=mean_profits.get)

            logger.info(f"Efficiency: {mean_eff:.1f}% ± {std_eff:.1f}%")
            logger.info(f"Top Trader: {top_trader}")

            result = {
                "environment": env_name,
                "efficiency_mean": mean_eff,
                "efficiency_std": std_eff,
                "top_trader": top_trader,
                "kaplan_rank": sorted(mean_profits, key=mean_profits.get, reverse=True).index(
                    "Kaplan"
                )
                + 1,
            }

            all_results.append(result)

        # Save results
        df = pd.DataFrame(all_results)
        output_file = self.output_dir / "phase3_full_tournament_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nPhase 3 results saved to {output_file}")

        # Calculate overall tournament metrics
        overall_efficiency = np.mean([r["efficiency_mean"] for r in all_results])
        logger.info("\n" + "=" * 60)
        logger.info("TOURNAMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Tournament Efficiency: {overall_efficiency:.1f}%")
        logger.info("Target (1994 paper): 89.7% ± 5%")

        if 84.7 <= overall_efficiency <= 94.7:
            logger.info("✅ SUCCESS: Efficiency within target range!")
        else:
            logger.info("⚠️  Efficiency outside target range")

        return all_results

    def run_all_phases(self):
        """Run all tournament phases sequentially."""
        logger.info("Starting Full Santa Fe Tournament Replication")
        logger.info("=" * 60)

        # Phase 1
        phase1_results = self.run_phase_1_baseline()

        # Phase 2
        phase2_results = self.run_phase_2_pairwise()

        # Phase 3
        phase3_results = self.run_phase_3_full_tournament()

        logger.info("\n" + "=" * 60)
        logger.info("TOURNAMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"All results saved to: {self.output_dir}")

        return {"phase1": phase1_results, "phase2": phase2_results, "phase3": phase3_results}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Santa Fe Tournament Replication")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which phase to run (1=baseline, 2=pairwise, 3=full, all=all phases)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    runner = FullTournamentRunner(output_dir=args.output)

    if args.phase == "1":
        runner.run_phase_1_baseline()
    elif args.phase == "2":
        runner.run_phase_2_pairwise()
    elif args.phase == "3":
        runner.run_phase_3_full_tournament()
    else:  # all
        runner.run_all_phases()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch Tournament Execution Script.

Runs all tournament experiments sequentially and aggregates results.

Usage:
    python scripts/run_tournament_batch.py [--output OUTPUT_DIR] [--categories CATEGORIES]

Examples:
    # Run all tournaments
    python scripts/run_tournament_batch.py

    # Run only pure and pairwise
    python scripts/run_tournament_batch.py --categories pure,pairwise

    # Custom output directory
    python scripts/run_tournament_batch.py --output results/my_tournament
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tournament_batch.log')
    ]
)
logger = logging.getLogger(__name__)


class TournamentBatchRunner:
    """Runs multiple tournament experiments in batch mode."""

    def __init__(self, output_dir: str = None, categories: List[str] = None):
        """
        Initialize batch runner.

        Args:
            output_dir: Base directory for saving results
            categories: List of tournament categories to run (default: all)
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"results/tournament_all_v_all_{self.timestamp}"
        self.categories = categories or ["pure", "pairwise", "mixed", "one_v_seven", "asymmetric"]

        self.results = []
        self.failed_experiments = []

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def discover_experiments(self) -> List[tuple]:
        """
        Discover all tournament experiment configs.

        Returns:
            List of (category, config_name) tuples
        """
        experiments = []
        base_path = Path("conf/experiment/tournament")

        for category in self.categories:
            category_path = base_path / category
            if not category_path.exists():
                logger.warning(f"Category directory not found: {category_path}")
                continue

            # Find all YAML files
            config_files = sorted(category_path.glob("*.yaml"))

            for config_file in config_files:
                # Remove .yaml extension
                config_name = config_file.stem
                experiments.append((category, config_name))

        logger.info(f"Discovered {len(experiments)} experiments across {len(self.categories)} categories")
        return experiments

    def run_single_experiment(self, category: str, config_name: str) -> dict:
        """
        Run a single tournament experiment.

        Args:
            category: Tournament category (pure, pairwise, etc.)
            config_name: Name of the config file (without .yaml)

        Returns:
            Dictionary with experiment results and metadata
        """
        logger.info(f"=" * 80)
        logger.info(f"Running: {category}/{config_name}")
        logger.info(f"=" * 80)

        start_time = time.time()

        try:
            # Setup output directory
            exp_output_dir = os.path.join(self.output_dir, category, config_name)
            os.makedirs(exp_output_dir, exist_ok=True)

            # Build command to run experiment
            cmd = [
                sys.executable,  # Use same Python interpreter
                "scripts/run_experiment.py",
                f"experiment=tournament/{category}/{config_name}",
                f"experiment.output_dir={exp_output_dir}"
            ]

            # Run experiment as subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per experiment
            )

            if result.returncode != 0:
                raise RuntimeError(f"Experiment failed with return code {result.returncode}\n{result.stderr}")

            # Load results CSV to calculate summary
            results_path = os.path.join(exp_output_dir, "results.csv")
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"Results CSV not found at {results_path}")

            results_df = pd.read_csv(results_path)

            # Calculate summary statistics
            summary = self._calculate_summary(results_df, category, config_name)

            elapsed_time = time.time() - start_time
            logger.info(f"✓ Completed in {elapsed_time:.1f}s - Efficiency: {summary['efficiency_mean']:.2f}%")

            return {
                "status": "success",
                "category": category,
                "config": config_name,
                "elapsed_time": elapsed_time,
                "output_path": exp_output_dir,
                **summary
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"✗ Failed after {elapsed_time:.1f}s: {str(e)}", exc_info=True)

            return {
                "status": "failed",
                "category": category,
                "config": config_name,
                "elapsed_time": elapsed_time,
                "error": str(e)
            }

    def _calculate_summary(self, results_df: pd.DataFrame, category: str, config_name: str) -> dict:
        """Calculate summary statistics from results."""
        # Overall efficiency (already in percentage in CSV)
        efficiency_mean = results_df['efficiency'].mean()
        efficiency_std = results_df['efficiency'].std()

        # Per-trader statistics (if available)
        trader_stats = {}
        if 'trader_type' in results_df.columns:
            for trader_type in results_df['trader_type'].unique():
                trader_df = results_df[results_df['trader_type'] == trader_type]
                trader_stats[trader_type] = {
                    'efficiency': trader_df['efficiency'].mean(),  # Already in percentage
                    'profit_mean': trader_df['profit'].mean() if 'profit' in trader_df.columns else None
                }

        return {
            "efficiency_mean": efficiency_mean,
            "efficiency_std": efficiency_std,
            "num_periods": len(results_df),
            "trader_stats": trader_stats
        }

    def run_all(self):
        """Run all discovered experiments."""
        experiments = self.discover_experiments()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING TOURNAMENT BATCH EXECUTION")
        logger.info(f"Total experiments: {len(experiments)}")
        logger.info(f"Categories: {', '.join(self.categories)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)
        logger.info("")

        overall_start = time.time()

        for idx, (category, config_name) in enumerate(experiments, 1):
            logger.info(f"\n[{idx}/{len(experiments)}] Running {category}/{config_name}...")

            result = self.run_single_experiment(category, config_name)
            self.results.append(result)

            if result["status"] == "failed":
                self.failed_experiments.append((category, config_name))

            # Periodic checkpoint
            if idx % 10 == 0:
                self._save_checkpoint(idx, len(experiments))

        overall_elapsed = time.time() - overall_start

        # Final summary
        self._generate_final_summary(overall_elapsed)

    def _save_checkpoint(self, completed: int, total: int):
        """Save intermediate checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{completed}of{total}.csv")
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(checkpoint_path, index=False)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _generate_final_summary(self, total_time: float):
        """Generate and save final summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TOURNAMENT BATCH EXECUTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful: {len([r for r in self.results if r['status'] == 'success'])}")
        logger.info(f"Failed: {len(self.failed_experiments)}")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info("=" * 80)

        if self.failed_experiments:
            logger.warning("\nFailed experiments:")
            for category, config in self.failed_experiments:
                logger.warning(f"  - {category}/{config}")

        # Save aggregate results
        results_df = pd.DataFrame(self.results)
        aggregate_path = os.path.join(self.output_dir, "aggregate_results.csv")
        results_df.to_csv(aggregate_path, index=False)
        logger.info(f"\nAggregate results saved to: {aggregate_path}")

        # Print efficiency summary by category
        logger.info("\nEfficiency by Category:")
        for category in self.categories:
            category_results = [r for r in self.results if r.get('category') == category and r['status'] == 'success']
            if category_results:
                efficiencies = [r['efficiency_mean'] for r in category_results]
                logger.info(f"  {category:15s}: {sum(efficiencies)/len(efficiencies):.2f}% (n={len(category_results)})")

        logger.info(f"\nFull results directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run tournament experiments in batch mode")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: results/tournament_all_v_all_TIMESTAMP)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to run (default: all)"
    )

    args = parser.parse_args()

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Run batch tournament
    runner = TournamentBatchRunner(output_dir=args.output, categories=categories)
    runner.run_all()


if __name__ == "__main__":
    main()

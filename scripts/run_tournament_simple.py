#!/usr/bin/env python3
"""
Simplified tournament runner using existing tournament infrastructure.
Runs the Santa Fe tournament replication experiments.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_tournament_batch import TournamentBatchRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Santa Fe Tournament Replication")
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Which categories to run (pure,pairwise,mixed,1v7,all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/tournament_replication_{timestamp}"
    else:
        output_dir = args.output

    # Parse categories
    if args.categories == "all":
        categories = ["pure", "pairwise", "mixed", "one_v_seven"]
    else:
        categories = args.categories.split(",")

    logger.info("=" * 60)
    logger.info("Santa Fe Tournament Replication")
    logger.info("=" * 60)
    logger.info(f"Categories: {categories}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Run tournament using existing batch runner
    runner = TournamentBatchRunner(
        output_dir=output_dir,
        categories=categories
    )

    # Execute tournament
    results = runner.run_all_tournaments()

    logger.info("\n" + "=" * 60)
    logger.info("TOURNAMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")

    # Print summary
    if results:
        logger.info("\nSummary:")
        for category, category_results in results.items():
            if category_results and "summary" in category_results:
                logger.info(f"\n{category}:")
                summary = category_results["summary"]
                if "efficiency_mean" in summary:
                    logger.info(f"  Mean efficiency: {summary['efficiency_mean']:.1f}%")
                if "efficiency_std" in summary:
                    logger.info(f"  Std deviation: {summary['efficiency_std']:.1f}%")


if __name__ == "__main__":
    main()
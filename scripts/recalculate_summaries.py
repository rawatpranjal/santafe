#!/usr/bin/env python3
"""
Recalculate tournament summaries with corrected efficiency calculation.
The CSVs have correct efficiency values, just the summaries were wrong.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recalculate_tournament_summaries():
    """Recalculate summaries for all completed tournaments."""

    # Find all tournament result directories
    result_dirs = sorted(Path("results").glob("tournament_all_v_all_*"))

    for result_dir in result_dirs:
        logger.info(f"\nProcessing: {result_dir}")

        # Process each category
        for category_dir in result_dir.iterdir():
            if category_dir.is_dir() and category_dir.name != "logs":
                logger.info(f"  Category: {category_dir.name}")

                all_results = []

                # Process each experiment in the category
                for exp_dir in sorted(category_dir.iterdir()):
                    if exp_dir.is_dir():
                        csv_file = exp_dir / "results.csv"
                        if csv_file.exists():
                            df = pd.read_csv(csv_file)

                            # Calculate correct summary
                            efficiency_mean = df['efficiency'].mean()
                            efficiency_std = df['efficiency'].std()

                            logger.info(f"    {exp_dir.name}: {efficiency_mean:.2f}% ± {efficiency_std:.2f}%")

                            all_results.append({
                                'experiment': exp_dir.name,
                                'efficiency_mean': efficiency_mean,
                                'efficiency_std': efficiency_std,
                                'num_periods': df['period'].nunique() if 'period' in df.columns else len(df)
                            })

                # Save corrected aggregate results
                if all_results:
                    aggregate_df = pd.DataFrame(all_results)
                    aggregate_file = result_dir / f"{category_dir.name}_corrected_summary.csv"
                    aggregate_df.to_csv(aggregate_file, index=False)

                    # Calculate overall statistics
                    overall_mean = aggregate_df['efficiency_mean'].mean()
                    overall_std = aggregate_df['efficiency_std'].mean()

                    logger.info(f"    Overall {category_dir.name}: {overall_mean:.2f}% ± {overall_std:.2f}%")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Recalculating Tournament Summaries")
    logger.info("=" * 60)

    recalculate_tournament_summaries()

    logger.info("\n" + "=" * 60)
    logger.info("Summary Recalculation Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
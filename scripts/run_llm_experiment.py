#!/usr/bin/env python3
"""
Run LLM agent experiments.

Usage:
  python scripts/run_llm_experiment.py --config placeholder_vs_zic
  python scripts/run_llm_experiment.py --config placeholder_vs_mixed
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
from engine.tournament import Tournament

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_llm_experiment(config_name: str):
    """
    Run an LLM experiment with the specified config.

    Args:
        config_name: Name of config file (without .yaml)
    """
    # Initialize Hydra
    hydra.initialize(config_path="../conf", version_base=None)

    # Load configuration
    cfg = hydra.compose(
        config_name="config",
        overrides=[f"experiment=llm/{config_name}"]
    )

    logger.info("=" * 60)
    logger.info(f"Running LLM Experiment: {cfg.experiment.name}")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Rounds: {cfg.experiment.num_rounds}")
    logger.info(f"  Periods: {cfg.market.num_periods}")
    logger.info(f"  Steps: {cfg.market.num_steps}")
    logger.info(f"  Tokens: {cfg.market.num_tokens}")
    logger.info(f"\n  Buyers:  {cfg.agents.buyer_types}")
    logger.info(f"  Sellers: {cfg.agents.seller_types}")
    logger.info("")

    # Create output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("llm_outputs/experiments") / f"{config_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inject output_dir into config for LLM agents
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.llm_output_dir = str(output_dir)
    OmegaConf.set_struct(cfg, True)

    # Run tournament
    tournament = Tournament(cfg)
    results_df = tournament.run()

    results_file = output_dir / "results.csv"
    results_df.to_csv(results_file, index=False)

    # Also save config for reference
    config_file = output_dir / "config.yaml"
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {results_file}")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Mean Efficiency: {results_df['efficiency'].mean():.2f}%")
    logger.info(f"  Std Efficiency: {results_df['efficiency'].std():.2f}%")

    # Get per-agent profit
    if 'profit_by_agent' in results_df.columns:
        logger.info("\n  Agent Profits (last round):")
        last_round = results_df.iloc[-1]
        # Parse profit dict if stored as string
        profit_str = last_round.get('profit_by_agent', '{}')
        logger.info(f"    {profit_str}")

    # Count PlaceholderLLM stats
    placeholder_count = len([a for a in cfg.agents.buyer_types if a == "PlaceholderLLM"])
    placeholder_count += len([a for a in cfg.agents.seller_types if a == "PlaceholderLLM"])
    logger.info(f"\n  PlaceholderLLM agents: {placeholder_count}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()

    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM agent experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name (e.g., 'placeholder_vs_zic', 'placeholder_vs_mixed')"
    )

    args = parser.parse_args()

    try:
        results = run_llm_experiment(args.config)
        logger.info("\nExperiment successful!")
        return 0
    except Exception as e:
        logger.error(f"\nExperiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

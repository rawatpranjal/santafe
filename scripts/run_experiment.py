"""
Run Experiment Script.

Usage:
    python scripts/run_experiment.py experiment=validation_zic
"""

import hydra
from omegaconf import DictConfig
from engine.tournament import Tournament
import os
import logging

# Configure logging
# Logging configured in main

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Configure logging
    log_level = getattr(logging, cfg.experiment.log_level.upper())
    
    # Force root logger
    logging.getLogger().setLevel(log_level)
    
    # Force specific loggers
    logging.getLogger("traders").setLevel(log_level)
    logging.getLogger("engine").setLevel(log_level)
    
    # Add handler if none exists (Hydra might capture, but we want stdout)
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
        logging.getLogger().addHandler(handler)
    
    logging.info(f"Running experiment: {cfg.experiment.name}")
    
    tournament = Tournament(cfg)
    results = tournament.run()
    
    # Save results
    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    
    logging.info(f"Results saved to {output_dir}")
    logging.info("Average Efficiency by Period:")
    print(results.groupby("period")["efficiency"].mean())

if __name__ == "__main__":
    main()

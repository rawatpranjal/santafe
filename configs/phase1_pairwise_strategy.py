# src_code/configs/phase1_pairwise_strategy.py
# Phase 1: Pairwise Strategy Comparison
# Tests each strategy against itself and others in isolated pairs
# Generates a competition matrix for strategy performance analysis

import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 500  # Sufficient for statistical significance
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 4

# Key strategies for pairwise comparison
# Focus on the main hierarchy: ZIC, ZIP, GD, Kaplan, RG
test_strategies = ["zic", "zip", "gd", "mgd", "kaplan", "rg"]

# This config will be used as a template for multiple runs
# Each run will test one buyer strategy vs one seller strategy

# --- BASE CONFIG Dictionary ---
BASE_CONFIG = {
    # --- Experiment Identification ---
    "experiment_name_prefix": "phase1_pairwise",  # Will be appended with strategy names
    "experiment_dir": "experiments/phase1/pairwise",
    
    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 5,   # 5 buyers of same type
    "num_sellers": 5,  # 5 sellers of same type
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # --- RNG Seeds ---
    "rng_seed_values": 3101,
    "rng_seed_auction": 3102,
    "rng_seed_rl": 3103,
    "log_level": "WARNING",  # Minimal logging for batch runs
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
    
    # --- Analysis Settings ---
    "save_detailed_stats": True,
    "save_trader_performance": True,
}

# Function to generate specific pairwise configs
def generate_pairwise_config(buyer_strategy, seller_strategy):
    """Generate a config for a specific buyer-seller strategy pair."""
    config = BASE_CONFIG.copy()
    config["experiment_name"] = f"phase1_pairwise_{buyer_strategy}_vs_{seller_strategy}"
    config["buyers"] = [{"type": buyer_strategy}] * config["num_buyers"]
    config["sellers"] = [{"type": seller_strategy}] * config["num_sellers"]
    config["rl_params"] = {}
    return config

# Generate all pairwise combinations
PAIRWISE_CONFIGS = []
for buyer_strat in test_strategies:
    for seller_strat in test_strategies:
        PAIRWISE_CONFIGS.append(generate_pairwise_config(buyer_strat, seller_strat))

# For compatibility with existing infrastructure
CONFIG = BASE_CONFIG.copy()
CONFIG["experiment_name"] = "phase1_pairwise_template"
CONFIG["buyers"] = [{"type": "zic"}] * 5  # Default placeholder
CONFIG["sellers"] = [{"type": "zic"}] * 5  # Default placeholder
CONFIG["rl_params"] = {}
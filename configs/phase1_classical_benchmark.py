# src_code/configs/phase1_classical_benchmark.py
# Phase 1: Classical Strategy Benchmark
# Validates the performance hierarchy: Kaplan/RG > ZIP/GD > ZIC
# Tests all classical strategies in a mixed market environment
# Standard SFI parameters: 4 Tokens, 25 Steps, 3 Periods

import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 1000  # Comprehensive evaluation
TRAINING_ROUNDS = 0  # No RL training
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25  # Standard SFI steps per period
NUM_PERIODS = 3        # Standard SFI periods per round
NUM_TOKENS = 4         # Standard tokens per agent per period

# All classical strategies to test
strategy_types = ["zic", "zip", "gd", "mgd", "el", "kaplan", "rg", "tt", "mu", "sk"]
num_strategies = len(strategy_types)

# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "phase1_classical_benchmark",
    "experiment_dir": "experiments/phase1",
    
    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": num_strategies,    # 10 Buyers (one of each type)
    "num_sellers": num_strategies,   # 10 Sellers (one of each type)
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,   # Standard SFI gametype for reproducible values
    
    # --- Trader Configuration ---
    # One of each strategy type per side for comprehensive comparison
    "buyers": [{"type": s_type} for s_type in strategy_types],
    "sellers": [{"type": s_type} for s_type in strategy_types],
    
    # --- PPO/RL Hyperparameters (Not used in Phase 1) ---
    "rl_params": {},
    
    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 3001,  # Phase 1 seed set
    "rng_seed_auction": 3002,
    "rng_seed_rl": 3003,      # Not used, but keep for consistency
    "log_level": "INFO",      # Moderate logging for long runs
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,  # Disable for 1000 rounds
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
    
    # --- Analysis Settings ---
    "save_detailed_stats": True,
    "save_trader_performance": True,
    "calculate_strategy_rankings": True,
}
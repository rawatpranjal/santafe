# code/configs/02a_fixed_strategy_mix_baseline_1k.py
# Baseline comparison of all fixed strategies.
# 9 Buyers, 9 Sellers (one of each type: ZIC, ZIP, GD, EL, Kaplan, TT, MU, SK, RG)
# Standard SFI parameters: 4 Tokens, 25 Steps, 3 Periods.
# Run for 1000 rounds.

import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 100 # Number of rounds (episodes)
TRAINING_ROUNDS = 0 # No RL training
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25 # Standard SFI steps per period
NUM_PERIODS = 3     # Standard SFI periods per round
NUM_TOKENS = 4      # Standard tokens per agent per period

# List of all fixed strategies to include
strategy_types = ["zic", "zip", "gd", "el", "kaplan", "tt", "mu", "sk", "rg"]
num_strategies = len(strategy_types)

# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "02a_fixed_strategy_mix_baseline_1k", # Will be overridden by runner
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": num_strategies,    # 9 Buyers
    "num_sellers": num_strategies,   # 9 Sellers
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,   # Standard SFI gametype

    # --- Trader Configuration ---
    # One of each fixed strategy type per side
    "buyers": [{"type": s_type} for s_type in strategy_types],
    "sellers": [{"type": s_type} for s_type in strategy_types],

    # --- PPO/RL Hyperparameters (Not used) ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2027, # Consistent seed set for 02 series
    "rng_seed_auction": 414,
    "rng_seed_rl": 525,      # Not used, but keep for consistency
    "log_level": "DEBUG",     # Console log level
    "log_level_rl": "WARNING", # RL logic logs (not applicable here)
    "log_to_file": True,
    "generate_per_round_plots": False, # Usually off for long runs
    "generate_eval_behavior_plots": False, # No RL agents to plot
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
}
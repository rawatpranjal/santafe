# code/configs/02b_fixed_strategy_mix_more_tokens_1k.py
# Comparison of all fixed strategies with MORE TOKENS.
# 9 Buyers, 9 Sellers (one of each type)
# Parameters: 8 Tokens, 25 Steps, 3 Periods.
# Run for 1000 rounds.

import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 1000
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 8 # Increased tokens

# List of all fixed strategies to include
strategy_types = ["zic", "zip", "gd", "el", "kaplan", "tt", "mu", "sk", "rg"]
num_strategies = len(strategy_types)

# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "02b_fixed_strategy_mix_more_tokens_1k",
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": num_strategies,    # 9 Buyers
    "num_sellers": num_strategies,   # 9 Sellers
    "num_tokens": NUM_TOKENS,        # Changed
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,

    # --- Trader Configuration ---
    "buyers": [{"type": s_type} for s_type in strategy_types],
    "sellers": [{"type": s_type} for s_type in strategy_types],

    # --- PPO/RL Hyperparameters (Not used) ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2027,
    "rng_seed_auction": 414,
    "rng_seed_rl": 525,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
}
# 02b_kaplan_mix_more_tokens_1k.py
# Kaplan performance replication - More Tokens (5B/5S, 8T, 25S)
# Uses Round=Episode structure

import random
import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 100
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3

# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "02b_kaplan_mix_more_tokens_1k", # Will be overridden by runner
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 5,
    "num_sellers": 5,
    "num_tokens": 8,    # CHANGED: More tokens per period
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,

    # --- Trader Configuration ---
    # Mix: 1 KP, 1 ZIP, 1 GD, 1 EL, 1 ZIC per side
    "buyers": [ {"type": "kaplan"}, {"type": "zip"}, {"type": "gd"}, {"type": "el"}, {"type": "zic"}, ],
    "sellers": [ {"type": "kaplan"}, {"type": "zip"}, {"type": "gd"}, {"type": "el"}, {"type": "zic"}, ],

    # --- PPO/RL Hyperparameters ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2027, # Keep same as baseline for comparison
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
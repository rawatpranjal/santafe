# 02c_kaplan_mix_double_agents_1k.py
# Kaplan performance replication - Double Agents (10B/10S, 4T, 25S)
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
    "experiment_name": "02c_kaplan_mix_double_agents_1k", # Will be overridden by runner
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 10,   # CHANGED: Double agents
    "num_sellers": 10,  # CHANGED: Double agents
    "num_tokens": 4,    # Standard tokens
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,

    # --- Trader Configuration ---
    # Mix: 2 KP, 2 ZIP, 2 GD, 2 EL, 2 ZIC per side (maintaining proportions)
    "buyers": [
        {"type": "kaplan"}, {"type": "kaplan"},
        {"type": "zip"}, {"type": "zip"},
        {"type": "gd"}, {"type": "gd"},
        {"type": "el"}, {"type": "el"},
        {"type": "zic"}, {"type": "zic"},
    ],
    "sellers": [
        {"type": "kaplan"}, {"type": "kaplan"},
        {"type": "zip"}, {"type": "zip"},
        {"type": "gd"}, {"type": "gd"},
        {"type": "el"}, {"type": "el"},
        {"type": "zic"}, {"type": "zic"},
    ],

    # --- PPO/RL Hyperparameters ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2027, # Keep same as baseline
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
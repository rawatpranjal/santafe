# 01e_baseline_zip_asymmetric_1k.py
# Run 5: ZIP Market Baseline (Asymmetric, 1000 Rounds)

import random
import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 1000
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3

# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "01e_baseline_zip_asymmetric_1k", # Matches filename base
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 6,    # Asymmetric
    "num_sellers": 4,   # Asymmetric
    "num_tokens": 4,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,

    # --- Trader Configuration ---
    "buyers": [{"type": "zip"}] * 6,
    "sellers": [{"type": "zip"}] * 4,

    # --- PPO Hyperparameters ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2025,
    "rng_seed_auction": 411,
    "rng_seed_rl": 522,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
}

# --- Sanity Check Prints (Optional) ---
# _exp_name = CONFIG.get("experiment_name", "default_exp")
# print(f"# Config File Loaded: {_exp_name}")
# print(f"# Total Rounds: {TOTAL_ROUNDS}")
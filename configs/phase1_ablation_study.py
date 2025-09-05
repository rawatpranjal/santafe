# src_code/configs/phase1_ablation_study.py
# Phase 1: Ablation Study - Information and Market Parameter Effects
# Tests the impact of different market conditions on strategy performance
# Varies: number of traders, tokens, market volatility

import numpy as np

# --- Base Parameters ---
TOTAL_ROUNDS = 300  # Moderate length for multiple conditions
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3

# Strategies to test in ablation
test_strategies = ["zic", "zip", "gd", "kaplan"]  # Representative strategies

# --- ABLATION CONDITIONS ---
# We'll create multiple configs testing different conditions

ABLATION_CONFIGS = []

# Condition 1: Vary number of traders (market thickness)
trader_counts = [2, 5, 10, 20]
for n_traders in trader_counts:
    config = {
        "experiment_name": f"phase1_ablation_traders_{n_traders}",
        "experiment_dir": "experiments/phase1/ablation",
        "num_rounds": TOTAL_ROUNDS,
        "num_periods": NUM_PERIODS,
        "num_steps": STEPS_PER_PERIOD,
        "num_training_rounds": TRAINING_ROUNDS,
        "num_buyers": n_traders,
        "num_sellers": n_traders,
        "num_tokens": 4,
        "min_price": 1,
        "max_price": 2000,
        "gametype": 6453,
        # Mix of strategies, repeated to fill trader count
        "buyers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(n_traders)],
        "sellers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(n_traders)],
        "rl_params": {},
        "rng_seed_values": 3201 + n_traders,
        "rng_seed_auction": 3301 + n_traders,
        "rng_seed_rl": 3401 + n_traders,
        "log_level": "WARNING",
        "log_level_rl": "WARNING",
        "log_to_file": True,
        "generate_per_round_plots": False,
        "generate_eval_behavior_plots": False,
        "num_eval_rounds_to_plot": 0,
        "save_rl_model": False,
        "load_rl_model_path": None,
        "save_detailed_stats": True,
    }
    ABLATION_CONFIGS.append(config)

# Condition 2: Vary number of tokens (trading capacity)
token_counts = [1, 2, 4, 8, 16]
for n_tokens in token_counts:
    config = {
        "experiment_name": f"phase1_ablation_tokens_{n_tokens}",
        "experiment_dir": "experiments/phase1/ablation",
        "num_rounds": TOTAL_ROUNDS,
        "num_periods": NUM_PERIODS,
        "num_steps": STEPS_PER_PERIOD,
        "num_training_rounds": TRAINING_ROUNDS,
        "num_buyers": 6,
        "num_sellers": 6,
        "num_tokens": n_tokens,
        "min_price": 1,
        "max_price": 2000,
        "gametype": 6453,
        # Mix of strategies
        "buyers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(6)],
        "sellers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(6)],
        "rl_params": {},
        "rng_seed_values": 3501 + n_tokens,
        "rng_seed_auction": 3601 + n_tokens,
        "rng_seed_rl": 3701 + n_tokens,
        "log_level": "WARNING",
        "log_level_rl": "WARNING",
        "log_to_file": True,
        "generate_per_round_plots": False,
        "generate_eval_behavior_plots": False,
        "num_eval_rounds_to_plot": 0,
        "save_rl_model": False,
        "load_rl_model_path": None,
        "save_detailed_stats": True,
    }
    ABLATION_CONFIGS.append(config)

# Condition 3: Vary price range (market volatility)
price_ranges = [(1, 500), (1, 1000), (1, 2000), (1, 5000)]
for min_p, max_p in price_ranges:
    config = {
        "experiment_name": f"phase1_ablation_range_{min_p}_{max_p}",
        "experiment_dir": "experiments/phase1/ablation",
        "num_rounds": TOTAL_ROUNDS,
        "num_periods": NUM_PERIODS,
        "num_steps": STEPS_PER_PERIOD,
        "num_training_rounds": TRAINING_ROUNDS,
        "num_buyers": 6,
        "num_sellers": 6,
        "num_tokens": 4,
        "min_price": min_p,
        "max_price": max_p,
        "gametype": 6453,
        # Mix of strategies
        "buyers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(6)],
        "sellers": [{"type": test_strategies[i % len(test_strategies)]} for i in range(6)],
        "rl_params": {},
        "rng_seed_values": 3801 + max_p // 100,
        "rng_seed_auction": 3901 + max_p // 100,
        "rng_seed_rl": 4001 + max_p // 100,
        "log_level": "WARNING",
        "log_level_rl": "WARNING",
        "log_to_file": True,
        "generate_per_round_plots": False,
        "generate_eval_behavior_plots": False,
        "num_eval_rounds_to_plot": 0,
        "save_rl_model": False,
        "load_rl_model_path": None,
        "save_detailed_stats": True,
    }
    ABLATION_CONFIGS.append(config)

# Default CONFIG for compatibility
CONFIG = ABLATION_CONFIGS[0] if ABLATION_CONFIGS else {
    "experiment_name": "phase1_ablation_default",
    "experiment_dir": "experiments/phase1/ablation",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 4,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    "buyers": [{"type": "zic"}] * 6,
    "sellers": [{"type": "zic"}] * 6,
    "rl_params": {},
    "rng_seed_values": 3200,
    "rng_seed_auction": 3300,
    "rng_seed_rl": 3400,
    "log_level": "WARNING",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
}
# src_code/configs/phase3_information_effects.py
# Phase 3: Information Effects Study
# Tests impact of different information levels on market dynamics
# Asymmetric information, learning under uncertainty

import numpy as np

# --- Parameters ---
TOTAL_ROUNDS = 3000
TRAINING_ROUNDS = 2700
EVALUATION_ROUNDS = 300

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 4

# Information level configurations for PPO agents
INFO_LEVELS = {
    "base": {
        "description": "Basic market information only",
        "state_components": ["own_values", "current_prices", "tokens_left"],
        "state_dim_multiplier": 1.0,
    },
    "extended": {
        "description": "Extended market information with history",
        "state_components": ["own_values", "current_prices", "tokens_left", 
                           "price_history", "volume_history", "spread"],
        "state_dim_multiplier": 1.5,
    },
    "full": {
        "description": "Full information including other agents' actions",
        "state_components": ["own_values", "current_prices", "tokens_left",
                           "price_history", "volume_history", "spread",
                           "other_agent_bids", "other_agent_asks", "market_depth"],
        "state_dim_multiplier": 2.0,
    },
    "privileged": {
        "description": "Privileged information (for testing)",
        "state_components": ["own_values", "current_prices", "tokens_left",
                           "price_history", "volume_history", "spread",
                           "other_agent_bids", "other_agent_asks", "market_depth",
                           "future_value_hints", "equilibrium_price"],
        "state_dim_multiplier": 2.5,
    }
}

# --- Information Asymmetry Experiments ---

# Experiment 1: Gradual information levels
CONFIG_GRADUAL_INFO = {
    "experiment_name": "phase3_info_gradual",
    "experiment_dir": "experiments/phase3/information",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 8,
    "num_sellers": 8,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Gradual distribution of information levels
    "buyers": [
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "privileged"},
    ],
    "sellers": [
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "privileged"},
    ],
    
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "INFO",
    },
    "rng_seed_values": 7001,
    "rng_seed_auction": 7002,
    "rng_seed_rl": 7003,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Information tracking
    "track_information_value": True,
    "calculate_info_advantage": True,
    "save_learning_curves_by_info": True,
}

# Experiment 2: Insider vs Outsider
CONFIG_INSIDER_TRADING = {
    "experiment_name": "phase3_info_insider",
    "experiment_dir": "experiments/phase3/information",
    "num_rounds": 2000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1800,
    "num_buyers": 8,
    "num_sellers": 8,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Few insiders with privileged info, many with basic
    "buyers": [
        {"type": "ppo_lstm", "information_level": "privileged"},  # 1 insider
        {"type": "ppo_lstm", "information_level": "base"},        # 7 outsiders
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
    ],
    "sellers": [
        {"type": "ppo_lstm", "information_level": "privileged"},  # 1 insider
        {"type": "ppo_lstm", "information_level": "base"},        # 7 outsiders
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
        {"type": "ppo_lstm", "information_level": "base"},
    ],
    
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "INFO",
    },
    "rng_seed_values": 7101,
    "rng_seed_auction": 7102,
    "rng_seed_rl": 7103,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Insider tracking
    "track_insider_profits": True,
    "measure_information_leakage": True,
    "detect_front_running": True,
}

# Experiment 3: Learning from limited information
CONFIG_LIMITED_INFO = {
    "experiment_name": "phase3_info_limited",
    "experiment_dir": "experiments/phase3/information",
    "num_rounds": 3000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 2700,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # All agents with minimal information
    "buyers": [{"type": "ppo_lstm", "information_level": "base"}] * 6,
    "sellers": [{"type": "ppo_lstm", "information_level": "base"}] * 6,
    
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.02,  # Higher entropy for exploration under uncertainty
        "max_grad_norm": 0.5,
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "INFO",
    },
    "rng_seed_values": 7201,
    "rng_seed_auction": 7202,
    "rng_seed_rl": 7203,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Learning tracking
    "track_learning_efficiency": True,
    "measure_exploration_patterns": True,
}

# Experiment 4: Dynamic information revelation
CONFIG_DYNAMIC_INFO = {
    "experiment_name": "phase3_info_dynamic",
    "experiment_dir": "experiments/phase3/information",
    "num_rounds": 4000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 3700,
    "num_buyers": 8,
    "num_sellers": 8,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Information levels change over time (simulated via config switches)
    "buyers": [
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 1000},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 1500},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 2000},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 2500},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "full"},
    ],
    "sellers": [
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 1000},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 1500},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 2000},
        {"type": "ppo_lstm", "information_level": "base", "upgrade_at_round": 2500},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "extended"},
        {"type": "ppo_lstm", "information_level": "full"},
        {"type": "ppo_lstm", "information_level": "full"},
    ],
    
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "INFO",
    },
    "rng_seed_values": 7301,
    "rng_seed_auction": 7302,
    "rng_seed_rl": 7303,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Dynamic tracking
    "track_adaptation_speed": True,
    "measure_strategy_shifts": True,
    "save_transition_points": True,
}

# Main configuration
CONFIG = {
    "experiment_name": "phase3_information_comprehensive",
    "experiment_dir": "experiments/phase3/information",
    "information_experiments": [
        CONFIG_GRADUAL_INFO,
        CONFIG_INSIDER_TRADING,
        CONFIG_LIMITED_INFO,
        CONFIG_DYNAMIC_INFO,
    ],
    
    # Default to gradual info for single run
    **CONFIG_GRADUAL_INFO,
}
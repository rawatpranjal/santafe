# src_code/configs/phase3_marl_dynamics.py
# Phase 3: Multi-Agent RL Dynamics
# Tests emergent behaviors with multiple PPO-LSTM agents
# Self-play, co-evolution, and market dynamics

import numpy as np

# --- MARL Parameters ---
TOTAL_ROUNDS = 10000  # Extended for emergent behavior study
TRAINING_ROUNDS = 9500
EVALUATION_ROUNDS = 500

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 4

# Shared PPO hyperparameters for all agents
PPO_MARL_PARAMS = {
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
    "information_level": "extended",
    "log_level_rl": "INFO",
    "log_training_stats": True,
    "save_checkpoints": True,
    "checkpoint_interval": 1000,
}

# --- MARL Experiment Configurations ---

# Experiment 1: Pure Self-Play (all PPO agents)
CONFIG_SELF_PLAY = {
    "experiment_name": "phase3_marl_self_play",
    "experiment_dir": "experiments/phase3/marl",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # All PPO agents
    "buyers": [{"type": "ppo_lstm"}] * 6,
    "sellers": [{"type": "ppo_lstm"}] * 6,
    
    "rl_params": PPO_MARL_PARAMS.copy(),
    "rng_seed_values": 6001,
    "rng_seed_auction": 6002,
    "rng_seed_rl": 6003,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 20,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # MARL specific settings
    "track_emergent_behaviors": True,
    "save_interaction_matrix": True,
    "calculate_nash_convergence": True,
}

# Experiment 2: PPO vs PPO with different information levels
CONFIG_INFO_ASYMMETRY = {
    "experiment_name": "phase3_marl_info_asymmetry",
    "experiment_dir": "experiments/phase3/marl",
    "num_rounds": 5000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 4500,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Mix of PPO agents with different information levels
    "buyers": [
        {"type": "ppo_lstm", "info_level": "base"},     # 2 with basic info
        {"type": "ppo_lstm", "info_level": "base"},
        {"type": "ppo_lstm", "info_level": "extended"}, # 2 with extended info
        {"type": "ppo_lstm", "info_level": "extended"},
        {"type": "ppo_lstm", "info_level": "full"},     # 2 with full info
        {"type": "ppo_lstm", "info_level": "full"},
    ],
    "sellers": [
        {"type": "ppo_lstm", "info_level": "base"},
        {"type": "ppo_lstm", "info_level": "base"},
        {"type": "ppo_lstm", "info_level": "extended"},
        {"type": "ppo_lstm", "info_level": "extended"},
        {"type": "ppo_lstm", "info_level": "full"},
        {"type": "ppo_lstm", "info_level": "full"},
    ],
    
    "rl_params": PPO_MARL_PARAMS.copy(),
    "rng_seed_values": 6101,
    "rng_seed_auction": 6102,
    "rng_seed_rl": 6103,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 20,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Track information advantage effects
    "track_info_advantage": True,
    "calculate_info_value": True,
}

# Experiment 3: Co-evolution (PPO agents evolving together with classical strategies)
CONFIG_COEVOLUTION = {
    "experiment_name": "phase3_marl_coevolution",
    "experiment_dir": "experiments/phase3/marl",
    "num_rounds": 8000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 7500,
    "num_buyers": 10,
    "num_sellers": 10,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Mix of PPO and adaptive classical strategies
    "buyers": (
        [{"type": "ppo_lstm"}] * 4 +      # 4 PPO agents
        [{"type": "zip"}] * 2 +           # 2 ZIP (adaptive)
        [{"type": "gd"}] * 2 +            # 2 GD (belief-based)
        [{"type": "kaplan"}] * 2          # 2 Kaplan (sophisticated)
    ),
    "sellers": (
        [{"type": "ppo_lstm"}] * 4 +
        [{"type": "zip"}] * 2 +
        [{"type": "gd"}] * 2 +
        [{"type": "kaplan"}] * 2
    ),
    
    "rl_params": PPO_MARL_PARAMS.copy(),
    "rng_seed_values": 6201,
    "rng_seed_auction": 6202,
    "rng_seed_rl": 6203,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 20,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Co-evolution tracking
    "track_strategy_evolution": True,
    "save_strategy_transitions": True,
}

# Experiment 4: Market manipulation detection
CONFIG_MANIPULATION = {
    "experiment_name": "phase3_marl_manipulation",
    "experiment_dir": "experiments/phase3/marl",
    "num_rounds": 3000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 2700,
    "num_buyers": 8,
    "num_sellers": 8,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Majority PPO with some naive agents (potential manipulation targets)
    "buyers": (
        [{"type": "ppo_lstm"}] * 5 +      # 5 PPO agents (potential manipulators)
        [{"type": "zic"}] * 3              # 3 ZIC agents (naive)
    ),
    "sellers": (
        [{"type": "ppo_lstm"}] * 5 +
        [{"type": "zic"}] * 3
    ),
    
    "rl_params": PPO_MARL_PARAMS.copy(),
    "rng_seed_values": 6301,
    "rng_seed_auction": 6302,
    "rng_seed_rl": 6303,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 20,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Manipulation detection
    "detect_price_manipulation": True,
    "track_collusion_metrics": True,
    "save_anomaly_patterns": True,
}

# Experiment 5: Scalability test (many agents)
CONFIG_SCALABILITY = {
    "experiment_name": "phase3_marl_scalability",
    "experiment_dir": "experiments/phase3/marl",
    "num_rounds": 2000,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1800,
    "num_buyers": 25,  # Large market
    "num_sellers": 25,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # Mix of many PPO agents with some classical
    "buyers": (
        [{"type": "ppo_lstm"}] * 15 +     # 15 PPO agents
        [{"type": "zip"}] * 5 +           # 5 ZIP
        [{"type": "gd"}] * 5              # 5 GD
    ),
    "sellers": (
        [{"type": "ppo_lstm"}] * 15 +
        [{"type": "zip"}] * 5 +
        [{"type": "gd"}] * 5
    ),
    
    "rl_params": {
        **PPO_MARL_PARAMS,
        "batch_size": 4096,  # Larger batch for more agents
    },
    "rng_seed_values": 6401,
    "rng_seed_auction": 6402,
    "rng_seed_rl": 6403,
    "log_level": "WARNING",  # Less logging for performance
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Performance tracking
    "track_computational_metrics": True,
    "measure_convergence_speed": True,
}

# Main MARL config
CONFIG = {
    "experiment_name": "phase3_marl_comprehensive",
    "experiment_dir": "experiments/phase3/marl",
    "marl_experiments": [
        CONFIG_SELF_PLAY,
        CONFIG_INFO_ASYMMETRY,
        CONFIG_COEVOLUTION,
        CONFIG_MANIPULATION,
        CONFIG_SCALABILITY,
    ],
    
    # Default to self-play for single run
    **CONFIG_SELF_PLAY,
}
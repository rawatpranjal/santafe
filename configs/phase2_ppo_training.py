# src_code/configs/phase2_ppo_training.py
# Phase 2: PPO-LSTM Agent Training
# Trains PPO-LSTM agents against classical strategies
# Progressive curriculum: ZIC -> ZIP -> GD -> Kaplan

import numpy as np

# --- Training Parameters ---
TOTAL_ROUNDS = 5000  # Extended training
TRAINING_ROUNDS = 4500  # Most rounds for training
EVALUATION_ROUNDS = 500  # Final evaluation

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 4

# --- PPO-LSTM Hyperparameters ---
PPO_HYPERPARAMS = {
    # Network architecture
    "nn_hidden_layers": [256, 128],  # MLP layers before LSTM
    "lstm_hidden_size": 128,         # LSTM hidden state size
    "lstm_num_layers": 2,             # Number of LSTM layers
    
    # Training parameters
    "learning_rate": 3e-4,
    "batch_size": 2048,               # Samples per update
    "n_epochs": 10,                   # PPO epochs per update
    "gamma": 0.99,                    # Discount factor
    "gae_lambda": 0.95,               # GAE lambda
    "clip_coef": 0.2,                 # PPO clip coefficient
    "value_loss_coef": 0.5,           # Value function loss weight
    "entropy_coef": 0.01,             # Entropy bonus
    "max_grad_norm": 0.5,             # Gradient clipping
    
    # Action space
    "num_price_actions": 21,          # Discretized price levels
    "price_range_pct": 0.15,          # Price action range (% of market)
    
    # Information level
    "information_level": "extended",  # Use extended market information
    
    # Logging
    "log_level_rl": "INFO",
    "log_training_stats": True,
    "save_checkpoints": True,
    "checkpoint_interval": 500,       # Save model every N rounds
}

# --- Training Curriculum Configs ---

# Stage 1: Train against ZIC (easiest opponent)
CONFIG_STAGE1 = {
    "experiment_name": "phase2_ppo_vs_zic",
    "experiment_dir": "experiments/phase2/training",
    "num_rounds": 1500,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1400,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 2 PPO buyers, 4 ZIC buyers vs 6 ZIC sellers
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": "zic"}] * 4,
    "sellers": [{"type": "zic"}] * 6,
    
    "rl_params": PPO_HYPERPARAMS.copy(),
    "rng_seed_values": 4001,
    "rng_seed_auction": 4002,
    "rng_seed_rl": 4003,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": None,
}

# Stage 2: Train against ZIP (intermediate)
CONFIG_STAGE2 = {
    "experiment_name": "phase2_ppo_vs_zip",
    "experiment_dir": "experiments/phase2/training",
    "num_rounds": 1500,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1400,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 2 PPO buyers, 4 ZIP buyers vs 6 ZIP sellers
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": "zip"}] * 4,
    "sellers": [{"type": "zip"}] * 6,
    
    "rl_params": PPO_HYPERPARAMS.copy(),
    "rng_seed_values": 4101,
    "rng_seed_auction": 4102,
    "rng_seed_rl": 4103,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": "experiments/phase2/training/phase2_ppo_vs_zic/models/final_model",  # Load from stage 1
}

# Stage 3: Train against GD/MGD (advanced)
CONFIG_STAGE3 = {
    "experiment_name": "phase2_ppo_vs_gd",
    "experiment_dir": "experiments/phase2/training",
    "num_rounds": 1500,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1400,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 2 PPO buyers, 2 GD + 2 MGD buyers vs 3 GD + 3 MGD sellers
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": "gd"}] * 2 + [{"type": "mgd"}] * 2,
    "sellers": [{"type": "gd"}] * 3 + [{"type": "mgd"}] * 3,
    
    "rl_params": PPO_HYPERPARAMS.copy(),
    "rng_seed_values": 4201,
    "rng_seed_auction": 4202,
    "rng_seed_rl": 4203,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": "experiments/phase2/training/phase2_ppo_vs_zip/models/final_model",  # Load from stage 2
}

# Stage 4: Train against Kaplan (hardest)
CONFIG_STAGE4 = {
    "experiment_name": "phase2_ppo_vs_kaplan",
    "experiment_dir": "experiments/phase2/training",
    "num_rounds": 2000,  # Longer training for hardest opponent
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 1900,
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 2 PPO buyers, 4 Kaplan buyers vs 3 Kaplan + 3 RG sellers
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": "kaplan"}] * 4,
    "sellers": [{"type": "kaplan"}] * 3 + [{"type": "rg"}] * 3,
    
    "rl_params": PPO_HYPERPARAMS.copy(),
    "rng_seed_values": 4301,
    "rng_seed_auction": 4302,
    "rng_seed_rl": 4303,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": True,
    "load_rl_model_path": "experiments/phase2/training/phase2_ppo_vs_gd/models/final_model",  # Load from stage 3
}

# Main training config (full curriculum)
CONFIG = {
    "experiment_name": "phase2_ppo_full_curriculum",
    "experiment_dir": "experiments/phase2/training",
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
    
    # Mixed market with PPO agents
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": "zip"}] * 2 + [{"type": "gd"}] * 2,
    "sellers": [{"type": "zip"}] * 2 + [{"type": "gd"}] * 2 + [{"type": "kaplan"}] * 2,
    
    "rl_params": PPO_HYPERPARAMS.copy(),
    "rng_seed_values": 4401,
    "rng_seed_auction": 4402,
    "rng_seed_rl": 4403,
    "log_level": "INFO",
    "log_level_rl": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 20,
    "save_rl_model": True,
    "load_rl_model_path": None,
    
    # Training stages for curriculum
    "curriculum_stages": [CONFIG_STAGE1, CONFIG_STAGE2, CONFIG_STAGE3, CONFIG_STAGE4],
}
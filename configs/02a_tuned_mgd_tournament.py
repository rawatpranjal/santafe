# Tuned MGD Tournament - Optimized parameters to maximize performance
# Testing MGD with longer memory and multi-unit optimization enabled

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "02a_tuned_mgd_tournament",
    "num_rounds": 3000,
    "num_periods": 3,
    "steps_per_period": 75,
    "train_rounds": 0,
    
    # Agent configuration - MGD vs baseline strategies
    "num_buyers": 8,
    "num_sellers": 8,
    
    # Tuned MGD buyers vs mixed competition
    "buyers": [
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},  # Longer memory
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},   # Medium memory
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},
        {"type": "zic"},  # Baseline competition
        {"type": "zic"},
        {"type": "gd"},   # GD competition
        {"type": "kaplan"}, # Kaplan competition
    ],
    
    # Tuned MGD sellers vs mixed competition
    "sellers": [
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "gd"},
        {"type": "kaplan"},
    ],
    
    # Market parameters - canonical SFI
    "min_price": 1,
    "max_price": 2000,
    "num_tokens": 4,
    
    # RL parameters (required)
    "rl_params": {
        "learning_rate": 0.0003,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "nn_hidden_layers": [128, 64],
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "use_reward_scaling": False,
        "log_training_stats": False,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "log_level_rl": "WARNING",
    },
    
    # Random seeds
    "rng_seed_values": 42,
    "rng_seed_auction": 43,
    "rng_seed_rl": 44,
    
    # Logging
    "log_level": "INFO",
    "log_to_file": True,
    "save_detailed_stats": True,
    "save_rl_model": False,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    
    # Game type
    "gametype": 6453,
}
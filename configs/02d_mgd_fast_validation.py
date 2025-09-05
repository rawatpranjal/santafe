# Fast MGD Validation - Efficient parameters for quick testing
# Testing if fixed MGD can outperform with reasonable computational cost

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "02d_mgd_fast_validation",
    "num_rounds": 1000,  # Shorter for quick validation
    "num_periods": 3,
    "steps_per_period": 50,  # Reduced from 75
    "train_rounds": 0,
    
    # Agent configuration - MGD vs key competitors
    "num_buyers": 6,  # Reduced for speed
    "num_sellers": 6,
    
    # Efficient MGD parameters
    "buyers": [
        {"type": "mgd", "history_len": 30, "use_multi_unit": True},  # Shorter memory
        {"type": "mgd", "history_len": 25, "use_multi_unit": True},
        {"type": "zic"},   # Direct competition
        {"type": "kaplan"}, # Strong competitor
        {"type": "gd"},    # GD baseline
        {"type": "el"},    # Current leader
    ],
    
    "sellers": [
        {"type": "mgd", "history_len": 30, "use_multi_unit": True},
        {"type": "mgd", "history_len": 25, "use_multi_unit": True},
        {"type": "zic"},
        {"type": "kaplan"},
        {"type": "gd"},
        {"type": "el"},
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
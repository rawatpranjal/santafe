# Quick Validation - Fast test of tuned strategies
# Short experiment to verify improvements quickly

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "02e_quick_validation",
    "num_rounds": 500,  # Very short for quick results
    "num_periods": 3,
    "steps_per_period": 40,  # Reduced steps
    "train_rounds": 0,
    
    # Agent configuration - key competitors only
    "num_buyers": 6,
    "num_sellers": 6,
    
    # Test optimized strategies vs strong baselines
    "buyers": [
        {"type": "mgd", "history_len": 25, "use_multi_unit": True},   # Fixed MGD
        {"type": "zip", "zip_beta": 0.45, "zip_gamma": 0.03, "zip_buyer_margin_low": -0.30, "zip_buyer_margin_high": -0.01},  # Tuned ZIP
        {"type": "el"},      # Current champion
        {"type": "kaplan"},  # Expected champion
        {"type": "gd"},      # Strong baseline
        {"type": "zic"},     # Control baseline
    ],
    
    "sellers": [
        {"type": "mgd", "history_len": 25, "use_multi_unit": True},
        {"type": "zip", "zip_beta": 0.45, "zip_gamma": 0.03, "zip_seller_margin_low": 0.01, "zip_seller_margin_high": 0.30},
        {"type": "el"},
        {"type": "kaplan"},
        {"type": "gd"},
        {"type": "zic"},
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
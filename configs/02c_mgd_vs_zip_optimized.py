# MGD vs ZIP Optimized Tournament
# Head-to-head comparison of tuned MGD and ZIP strategies

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "02c_mgd_vs_zip_optimized",
    "num_rounds": 4000,  # More rounds for statistical significance
    "num_periods": 3,
    "steps_per_period": 75,
    "train_rounds": 0,
    
    # Agent configuration - Tuned MGD vs Tuned ZIP
    "num_buyers": 8,
    "num_sellers": 8,
    
    # Mixed optimized buyers
    "buyers": [
        # Optimized MGD buyers
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        # Optimized ZIP buyers
        {"type": "zip", "zip_beta": 0.45, "zip_gamma": 0.03, "zip_buyer_margin_low": -0.30, "zip_buyer_margin_high": -0.01},
        {"type": "zip", "zip_beta": 0.40, "zip_gamma": 0.05, "zip_buyer_margin_low": -0.25, "zip_buyer_margin_high": -0.02},
        {"type": "zip", "zip_beta": 0.50, "zip_gamma": 0.02, "zip_buyer_margin_low": -0.35, "zip_buyer_margin_high": -0.05},
        # Baseline controls
        {"type": "zic"},
        {"type": "gd"},
    ],
    
    # Mixed optimized sellers
    "sellers": [
        # Optimized MGD sellers
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        {"type": "mgd", "history_len": 75, "use_multi_unit": True},
        {"type": "mgd", "history_len": 100, "use_multi_unit": True},
        # Optimized ZIP sellers
        {"type": "zip", "zip_beta": 0.45, "zip_gamma": 0.03, "zip_seller_margin_low": 0.01, "zip_seller_margin_high": 0.30},
        {"type": "zip", "zip_beta": 0.40, "zip_gamma": 0.05, "zip_seller_margin_low": 0.02, "zip_seller_margin_high": 0.25},
        {"type": "zip", "zip_beta": 0.50, "zip_gamma": 0.02, "zip_seller_margin_low": 0.05, "zip_seller_margin_high": 0.35},
        # Baseline controls
        {"type": "zic"},
        {"type": "gd"},
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
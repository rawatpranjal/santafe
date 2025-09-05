CONFIG = {
    # Experiment name  
    "experiment_name": "test_ppo_full_range",
    
    # Basic auction parameters
    "num_rounds": 1500,
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 1200,
    
    # Market structure - 1 PPO + 3 ZIC buyers vs 4 ZIC sellers
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # 1 PPO with full range + 3 ZIC buyers
    "buyers": [{"type": "ppo_handcrafted"}] + [{"type": "zic"}] * 3,
    "sellers": [{"type": "zic"}] * 4,
    
    # PPO with FULL market access
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128, 
        "lstm_num_layers": 2,
        
        # Training parameters
        "learning_rate": 5e-4,  # Moderate LR for stable learning
        "batch_size": 1024,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.05,  # Higher entropy to explore full range
        "max_grad_norm": 0.5,
        
        # Action space - MANY granular actions across full market
        "num_price_actions": 51,  # 50 price levels + 1 pass action
        "price_range_pct": 1.0,  # Not used with new mapping, but kept for compatibility
        
        # Reward
        "use_reward_scaling": False,  # Direct profit signal
        
        # Logging
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 42,
    "rng_seed_auction": 43,
    "rng_seed_rl": 44,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
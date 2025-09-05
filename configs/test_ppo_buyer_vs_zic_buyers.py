CONFIG = {
    # Experiment name - will be used for output directory
    "experiment_name": "test_ppo_buyer_vs_zic_buyers",
    
    # Basic auction parameters
    "num_rounds": 1000,  # Total rounds
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 800,  # Training rounds before evaluation
    
    # Market structure - 1 PPO + 3 ZIC buyers vs 4 ZIC sellers
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # 1 PPO buyer + 3 ZIC buyers vs 4 ZIC sellers
    "buyers": [{"type": "ppo_handcrafted"}] + [{"type": "zic"}] * 3,
    "sellers": [{"type": "zic"}] * 4,
    
    # PPO parameters - tuned for competitive environment
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [128, 64],
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        
        # Training parameters
        "learning_rate": 3e-4,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        
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
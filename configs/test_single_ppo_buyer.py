CONFIG = {
    # Experiment name - will be used for output directory
    "experiment_name": "test_single_ppo_buyer",
    
    # Basic auction parameters
    "num_rounds": 1000,  # Total rounds
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 800,  # Training rounds before evaluation
    
    # Market structure - SINGLE PPO BUYER ONLY
    "num_buyers": 1,
    "num_sellers": 5,  # Multiple ZIC sellers for liquidity
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Single PPO buyer vs multiple ZIC sellers
    "buyers": [{"type": "ppo_handcrafted"}],
    "sellers": [{"type": "zic"}] * 5,  # 5 ZIC sellers
    
    # PPO parameters - tuned for single agent learning
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [128, 64],  # Smaller network for single agent
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        
        # Training parameters
        "learning_rate": 3e-4,  # Higher LR for faster single-agent learning
        "batch_size": 512,  # Smaller batch for single agent
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,  # Lower entropy for focused learning
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 21,  # Moderate action space
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
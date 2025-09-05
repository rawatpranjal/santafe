CONFIG = {
    # Experiment name - will be used for output directory
    "experiment_name": "test_ppo_buyer_improved",
    
    # Basic auction parameters
    "num_rounds": 1500,  # More training rounds
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 1200,  # Extended training
    
    # Market structure - 1 PPO + 3 ZIC buyers vs 4 ZIC sellers
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # 1 improved PPO buyer + 3 ZIC buyers vs 4 ZIC sellers
    "buyers": [{"type": "ppo_handcrafted"}] + [{"type": "zic"}] * 3,
    "sellers": [{"type": "zic"}] * 4,
    
    # Improved PPO parameters
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],  # Larger network
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        
        # Training parameters - tuned for competition
        "learning_rate": 1e-3,  # Higher learning rate
        "batch_size": 1024,  # Larger batch
        "n_epochs": 15,  # More epochs per update
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.03,  # Higher entropy for exploration
        "max_grad_norm": 0.5,
        
        # Action space - CRITICAL CHANGE
        "num_price_actions": 31,  # More granular actions
        "price_range_pct": 0.40,  # MUCH wider range for aggressive bidding
        
        # Reward shaping
        "use_reward_scaling": False,  # Disable normalization for direct profit signal
        
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
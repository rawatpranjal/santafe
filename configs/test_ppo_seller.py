# Test PPO as seller against ZIC sellers
# PPO agents often perform better as sellers due to the market dynamics

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "test_ppo_seller",
    "num_rounds": 1500,
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 1200,  # Train for 80% of rounds
    
    # Agent configuration - PPO as seller
    "num_buyers": 4,
    "num_sellers": 4,
    
    # All buyers are ZIC
    "buyers": [
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
    ],
    
    # Mix of PPO and ZIC sellers
    "sellers": [
        {"type": "ppo_handcrafted"},  # Our PPO seller
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
    ],
    
    # Market parameters
    "min_price": 50,
    "max_price": 250,
    "num_tokens": 3,
    
    # RL parameters - optimized for seller role
    "rl_params": {
        "learning_rate": 0.0003,
        "batch_size": 1024,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.03,  # Moderate exploration
        "max_grad_norm": 0.5,
        "nn_hidden_layers": [256, 128],
        "num_price_actions": 51,  # Good granularity
        "price_range_pct": 0.5,  # Moderate price range
        "use_reward_scaling": False,
        "log_training_stats": True,
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "log_level_rl": "INFO",
    },
    
    # Random seeds for reproducibility
    "rng_seed_values": 42,
    "rng_seed_auction": 43,
    "rng_seed_rl": 44,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "save_detailed_stats": True,
    "save_rl_model": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    
    # Game type (453 = standard Santa Fe parameters)
    "gametype": 453,
}
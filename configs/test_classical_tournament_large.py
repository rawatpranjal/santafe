# Large-scale classical agents tournament
# Testing all classical strategies with more agents and rounds for statistical significance
# 10v10 agents, 5000 rounds, all strategies represented

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "test_classical_tournament_large",
    "num_rounds": 5000,  # Increased from 2000 for better statistics
    "num_periods": 3,
    "steps_per_period": 25,  # Slightly more steps per period
    "train_rounds": 0,  # No training for classical agents
    
    # Agent configuration - 10v10 for larger market
    "num_buyers": 10,
    "num_sellers": 10,
    
    # Mixed buyers - balanced representation of all strategies
    "buyers": [
        {"type": "zic"},   # 2 ZIC buyers (baseline)
        {"type": "zic"},
        {"type": "zip"},   # 2 ZIP buyers
        {"type": "zip"},
        {"type": "kaplan"}, # 2 Kaplan buyers
        {"type": "kaplan"},
        {"type": "gd"},    # 2 GD buyers
        {"type": "gd"},
        {"type": "mgd"},   # 1 MGD buyer
        {"type": "el"},    # 1 EL buyer
    ],
    
    # Mixed sellers - balanced representation of all strategies
    "sellers": [
        {"type": "zic"},   # 2 ZIC sellers (baseline)
        {"type": "zic"},
        {"type": "zip"},   # 2 ZIP sellers
        {"type": "zip"},
        {"type": "kaplan"}, # 2 Kaplan sellers
        {"type": "kaplan"},
        {"type": "gd"},    # 2 GD sellers
        {"type": "gd"},
        {"type": "mgd"},   # 1 MGD seller
        {"type": "el"},    # 1 EL seller
    ],
    
    # Market parameters
    "min_price": 50,
    "max_price": 250,
    "num_tokens": 3,
    
    # RL parameters (not used but needed for config)
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
    
    # Random seeds for reproducibility
    "rng_seed_values": 42,
    "rng_seed_auction": 43,
    "rng_seed_rl": 44,
    
    # Logging and output - set to DEBUG to catch any issues
    "log_level": "INFO",
    "log_to_file": True,
    "save_detailed_stats": True,
    "save_rl_model": False,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    
    # Game type (453 = standard Santa Fe parameters)
    "gametype": 453,
}
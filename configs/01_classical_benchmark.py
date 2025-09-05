# Canonical SFI Classical Benchmark - Exact replication of original Santa Fe parameters
# This is the foundation experiment for validating our implementation against the literature
# Using exact parameters from the original Santa Fe Institute double auction experiments

CONFIG = {
    # Basic experiment parameters - matching original SFI setup
    "experiment_name": "01_classical_benchmark",
    "num_rounds": 2000,  # Standard SFI benchmark
    "num_periods": 3,    # Three trading periods per round
    "steps_per_period": 75,  # Full 75 steps as per original
    "train_rounds": 0,  # No training for classical agents
    
    # Agent configuration - 6v6 balanced market
    "num_buyers": 6,
    "num_sellers": 6,
    
    # Pure strategy populations for clean baseline
    "buyers": [
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
    ],
    
    "sellers": [
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
        {"type": "zic"},
    ],
    
    # Market parameters - EXACT SFI specification
    "min_price": 1,      # Original SFI range
    "max_price": 2000,   # Original SFI range
    "num_tokens": 4,     # 4 tokens per agent as per SFI
    
    # RL parameters (not used but required by config structure)
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
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "save_detailed_stats": True,
    "save_rl_model": False,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    
    # Game type - CANONICAL SFI PARAMETER
    "gametype": 6453,  # Original SFI value distribution
}
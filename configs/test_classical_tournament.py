# Classical agents tournament - testing Kaplan, ZIP, GD against ZIC baseline
# This will establish proper baselines for the paper

CONFIG = {
    # Basic experiment parameters
    "experiment_name": "test_classical_tournament",
    "num_rounds": 2000,
    "num_periods": 3,
    "steps_per_period": 20,
    "train_rounds": 0,  # No training for classical agents
    
    # Agent configuration - diverse mix of classical strategies
    "num_buyers": 6,
    "num_sellers": 6,
    
    # Mixed buyers - 2 each of top strategies
    "buyers": [
        {"type": "kaplan"},
        {"type": "kaplan"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "gd"},
        {"type": "zic"},
    ],
    
    # Mixed sellers - 2 each of top strategies
    "sellers": [
        {"type": "kaplan"},
        {"type": "kaplan"},
        {"type": "zip"},
        {"type": "zip"},
        {"type": "gd"},
        {"type": "zic"},
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
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "save_detailed_stats": True,
    "save_rl_model": False,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    
    # Game type (453 = standard Santa Fe parameters)
    "gametype": 453,
}
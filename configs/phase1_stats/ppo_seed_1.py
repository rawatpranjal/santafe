# Phase 1: Statistical Validation - Seed 1
"""
Replicate PPO performance with different random seeds.
Base configuration matches the successful single_ppo_vs_mixed test.
"""

CONFIG = {
    "experiment_name": "ppo_statistical_seed1",
    "experiment_dir": "results/active/phase1_seed1",
    "num_rounds": 1000,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 800,
    
    # Market setup (same as baseline)
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Same agent mix as successful test
    "buyers": [
        {"type": "ppo_handcrafted"},
        {"type": "zic"},
        {"type": "zip"},
        {"type": "zi"}
    ],
    "sellers": [
        {"type": "ppo_handcrafted"},
        {"type": "zic"},
        {"type": "zip"},
        {"type": "zi"}
    ],
    
    # Proven PPO parameters
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.05,
        "max_grad_norm": 0.5,
        "use_lr_annealing": True,
        "use_entropy_annealing": True,
        "use_reward_scaling": True,
        "target_kl": 0.02,
        "norm_adv": True,
        "clip_vloss": True,
        "num_price_actions": 41,
        "price_range_pct": 0.20,
        "epsilon_greedy": 0.10,
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Different seeds for statistical validation
    "rng_seed_values": 1001,
    "rng_seed_auction": 1002,
    "rng_seed_rl": 1003,
    
    # Logging
    "log_level": "INFO",
    "log_to_file": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
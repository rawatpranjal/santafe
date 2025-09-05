# configs/test_ppo_mixed_strategy.py
"""
Mixed strategy training for PPO: Train against diverse opponents.
Uses 2 PPO + 2 ZIP + 2 ZIC on each side for varied market dynamics.
"""

CONFIG = {
    "experiment_name": "test_ppo_mixed_strategy",
    "experiment_dir": "experiments/ppo_mixed",
    "num_rounds": 500,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 400,  # 400 training, 100 evaluation
    
    # Market setup
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Mixed strategy: 2 PPO + 2 ZIP + 2 ZIC on each side
    "buyers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zip"}] * 2 + [{"type": "zic"}] * 2,
    "sellers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zip"}] * 2 + [{"type": "zic"}] * 2,
    
    # PPO parameters - competitive settings
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        
        # Competitive hyperparameters
        "learning_rate": 1e-4,  # Stable learning
        "batch_size": 4096,     # Large batch for stability
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.04,   # High entropy for exploration vs diverse opponents
        "max_grad_norm": 0.5,
        
        # Optimization features
        "use_lr_annealing": True,
        "use_entropy_annealing": False,  # Keep high entropy
        "use_reward_scaling": True,
        "target_kl": 0.025,  # Higher tolerance
        "norm_adv": True,
        "clip_vloss": True,
        
        # Action space
        "num_price_actions": 41,
        "price_range_pct": 0.20,
        
        # Exploration
        "epsilon_greedy": 0.10,  # Higher exploration vs diverse opponents
        
        # Logging
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
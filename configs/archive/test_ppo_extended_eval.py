# configs/test_ppo_extended_eval.py
"""
Extended evaluation for PPO: Longer training with objective performance tracking.
1500 training rounds + 500 evaluation rounds for neutral analysis.
"""

CONFIG = {
    "experiment_name": "test_ppo_extended_eval",
    "experiment_dir": "experiments/ppo_extended_eval",
    "num_rounds": 2000,  # Extended for objective evaluation
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 1500,  # 1500 training, 500 evaluation
    
    # Market setup
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Balanced opponent mix for objective evaluation
    "buyers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    "sellers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    
    # PPO parameters - stable configuration
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        
        # Conservative hyperparameters for objective evaluation
        "learning_rate": 1e-4,
        "batch_size": 4096,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.02,  # Moderate entropy
        "max_grad_norm": 0.5,
        
        # Optimization features
        "use_lr_annealing": True,
        "use_entropy_annealing": False,
        "use_reward_scaling": True,
        "target_kl": 0.02,
        "norm_adv": True,
        "clip_vloss": True,
        
        # Action space
        "num_price_actions": 41,
        "price_range_pct": 0.20,
        
        # Exploration
        "epsilon_greedy": 0.05,
        
        # Logging
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Enhanced logging for analysis
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
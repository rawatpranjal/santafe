# configs/single_ppo_vs_mixed.py
"""
Single PPO agent test: Can one PPO agent outperform a mix of trading opponents?
Testing 1 PPO buyer + 1 PPO seller against diverse opponent mix.
"""

CONFIG = {
    "experiment_name": "single_ppo_vs_mixed",
    "experiment_dir": "experiments/single_ppo_vs_mixed",
    "num_rounds": 1000,  # Sufficient rounds to see learning
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 800,  # 800 training, 200 evaluation
    
    # Market setup - asymmetric to test PPO adaptability
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Single PPO against mixed opponents
    "buyers": [
        {"type": "ppo_handcrafted"},  # 1 PPO buyer
        {"type": "zic"},              # 1 ZIC buyer
        {"type": "zip"},              # 1 ZIP buyer
        {"type": "zi"}                # 1 ZI buyer (baseline)
    ],
    "sellers": [
        {"type": "ppo_handcrafted"},  # 1 PPO seller
        {"type": "zic"},              # 1 ZIC seller
        {"type": "zip"},              # 1 ZIP seller
        {"type": "zi"}                # 1 ZI seller (baseline)
    ],
    
    # Optimized PPO parameters for competitive performance
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        
        # Hyperparameters tuned for competitive trading
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.05,  # Higher entropy for exploration against diverse opponents
        "max_grad_norm": 0.5,
        
        # Optimization features
        "use_lr_annealing": True,
        "use_entropy_annealing": True,
        "use_reward_scaling": True,
        "target_kl": 0.02,
        "norm_adv": True,
        "clip_vloss": True,
        
        # Action space
        "num_price_actions": 41,
        "price_range_pct": 0.20,
        
        # Exploration against diverse strategies
        "epsilon_greedy": 0.10,  # 10% random exploration
        
        # Logging
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Enhanced logging for individual agent analysis
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
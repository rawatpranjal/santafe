# configs/test_ppo_vs_zic_long.py
"""
Extended test: PPOHandcrafted vs ZIC with sufficient training time.
Using wider price range to ensure trading opportunities.
"""

CONFIG = {
    "experiment_name": "test_ppo_vs_zic_long",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 1500,
    "num_periods": 3,
    "num_steps": 25,
    "num_training_rounds": 1300,  # 1300 training, 200 evaluation
    
    # Market setup - wider price range for more opportunities
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,  # Try different gametype
    
    # Agent composition: 4 PPOHandcrafted + 2 ZIC buyers, opposite for sellers
    "buyers": [{"type": "ppo_handcrafted"}] * 4 + [{"type": "zic"}] * 2,
    "sellers": [{"type": "zic"}] * 4 + [{"type": "ppo_handcrafted"}] * 2,
    
    # PPO parameters - tuned for better learning
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        
        # Training parameters - adjusted for better learning
        "learning_rate": 1e-4,  # Lower for stability
        "batch_size": 2048,  # Larger batch for stability
        "n_epochs": 10,
        "gamma": 0.995,  # Higher for long-term rewards
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.005,  # Lower entropy for exploitation
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 31,  # More actions for finer control
        "price_range_pct": 0.25,  # Wider exploration range
        
        # Logging
        "log_level_rl": "WARNING",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
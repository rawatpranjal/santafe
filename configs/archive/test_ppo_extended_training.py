# configs/test_ppo_extended_training.py
"""
Extended training test for PPO Handcrafted agents.
2500 rounds total to see if performance improves with more training.
"""

CONFIG = {
    "experiment_name": "test_ppo_extended_training",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 2500,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 2000,  # 2000 training, 500 evaluation
    
    # Market setup
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Agent composition: 3 PPOHandcrafted + 3 ZIC on each side for balance
    "buyers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    "sellers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    
    # PPO parameters - tuned for better learning
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        
        # Training parameters - adjusted for stability
        "learning_rate": 1e-4,  # Lower for stability over long training
        "batch_size": 1024,
        "n_epochs": 10,
        "gamma": 0.995,  # Higher for long-term rewards
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.02,  # Higher for more exploration
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 31,  # More actions for finer control
        "price_range_pct": 0.20,  # Wider range for exploration
        
        # Logging
        "log_level_rl": "WARNING",
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
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
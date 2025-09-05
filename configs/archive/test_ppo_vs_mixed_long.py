# configs/test_ppo_vs_mixed_long.py
"""
Extended test: PPOHandcrafted vs mixed market of different trader types.
This tests generalization ability.
"""

CONFIG = {
    "experiment_name": "test_ppo_vs_mixed_long",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 1500,
    "num_periods": 3,
    "num_steps": 25,
    "num_training_rounds": 1300,
    
    # Market setup
    "num_buyers": 8,  # More agents for diverse market
    "num_sellers": 8,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Agent composition: Mixed market
    # Buyers: 4 PPOHandcrafted, 1 ZIC, 1 ZIP, 1 GD, 1 Kaplan
    "buyers": [{"type": "ppo_handcrafted"}] * 4 + [{"type": "zic"}] + [{"type": "zip"}] + [{"type": "gd"}] + [{"type": "kaplan"}],
    # Sellers: 4 PPOHandcrafted, 1 ZIC, 1 ZIP, 1 GD, 1 Kaplan  
    "sellers": [{"type": "ppo_handcrafted"}] * 4 + [{"type": "zic"}] + [{"type": "zip"}] + [{"type": "gd"}] + [{"type": "kaplan"}],
    
    # PPO parameters
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        
        # Training parameters
        "learning_rate": 1e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,  # Higher entropy for diverse opponents
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 31,
        "price_range_pct": 0.25,
        
        # Logging
        "log_level_rl": "WARNING",
        "log_training_stats": True,
    },
    
    # Seeds
    "rng_seed_values": 6024,
    "rng_seed_auction": 6025,
    "rng_seed_rl": 6026,
    
    # Output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
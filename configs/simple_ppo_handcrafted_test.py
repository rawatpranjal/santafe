# configs/simple_ppo_handcrafted_test.py
"""
Simple test with good market conditions to verify PPOHandcrafted works.
Using gametype 0 (fixed values) for predictable surplus.
"""

CONFIG = {
    "experiment_name": "simple_ppo_handcrafted_test",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 200,
    "num_periods": 2,
    "num_steps": 15,
    "num_training_rounds": 150,
    
    # Market setup - simpler configuration
    "num_buyers": 4,
    "num_sellers": 4,
    "num_tokens": 2,
    "min_price": 50,
    "max_price": 150,
    "gametype": 0,  # Fixed values for predictable surplus
    
    # Agent composition: 2 PPOHandcrafted vs 2 ZIC on each side
    "buyers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zic"}] * 2,
    "sellers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zic"}] * 2,
    
    # PPO parameters
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [128, 64],
        "lstm_hidden_size": 64,
        "lstm_num_layers": 1,
        
        # Training parameters
        "learning_rate": 5e-4,  # Slightly higher for faster learning
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 21,
        "price_range_pct": 0.2,  # Wider range for exploration
        
        # Logging
        "log_level_rl": "WARNING",
        "log_training_stats": True,
    },
    
    # Seeds
    "rng_seed_values": 12345,
    "rng_seed_auction": 12346,
    "rng_seed_rl": 12347,
    
    # Output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
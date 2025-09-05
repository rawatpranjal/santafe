# configs/test_ppo_vs_kaplan_long.py
"""
Extended test: PPOHandcrafted vs Kaplan with sufficient training time.
Kaplan is the strongest heuristic - this is the ultimate test.
"""

CONFIG = {
    "experiment_name": "test_ppo_vs_kaplan_long",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 2000,  # Extra training for hardest opponent
    "num_periods": 3,
    "num_steps": 25,
    "num_training_rounds": 1800,  # 1800 training, 200 evaluation
    
    # Market setup
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Agent composition: 4 PPOHandcrafted + 2 Kaplan buyers, opposite for sellers
    "buyers": [{"type": "ppo_handcrafted"}] * 4 + [{"type": "kaplan"}] * 2,
    "sellers": [{"type": "kaplan"}] * 4 + [{"type": "ppo_handcrafted"}] * 2,
    
    # PPO parameters - extra tuning for Kaplan
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
        "entropy_coef": 0.003,  # Even lower for exploitation against Kaplan
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 31,
        "price_range_pct": 0.25,
        
        # Logging
        "log_level_rl": "WARNING",
        "log_training_stats": True,
    },
    
    # Seeds
    "rng_seed_values": 5024,
    "rng_seed_auction": 5025,
    "rng_seed_rl": 5026,
    
    # Output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
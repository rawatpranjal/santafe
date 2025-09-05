# configs/test_ppo_handcrafted_vs_zip.py
"""
The real test: PPOHandcrafted vs ZIP agents
ZIP is a strong adaptive agent. If PPOHandcrafted beats ZIP, we have proof it works.
"""

# Test parameters
TOTAL_ROUNDS = 1000
TRAINING_ROUNDS = 900
EVALUATION_ROUNDS = 100

CONFIG = {
    "experiment_name": "test_ppo_handcrafted_vs_zip",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": 3,
    "num_steps": 25,
    "num_training_rounds": TRAINING_ROUNDS,
    
    # Market setup - wider price range for real trading opportunities
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,  # Changed to gametype with actual profit opportunities
    
    # Agent composition: 2 PPOHandcrafted + 4 ZIP buyers vs 6 ZIP sellers
    "buyers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zip"}] * 4,
    "sellers": [{"type": "zip"}] * 6,
    
    # PPO parameters - tuned for better performance
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        
        # Training parameters
        "learning_rate": 3e-4,
        "batch_size": 1024,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        
        # Action space
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        
        # Logging
        "log_level_rl": "WARNING",
        "log_training_stats": True,
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 100,
    "rng_seed_auction": 101,
    "rng_seed_rl": 102,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
    
    # Optional: Load model from ZIC training as warm start
    # "load_rl_model_path": "experiments/ppo_handcrafted_tests/test_ppo_handcrafted_vs_zic/models/final_model",
}
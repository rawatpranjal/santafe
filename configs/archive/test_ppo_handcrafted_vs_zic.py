# configs/test_ppo_handcrafted_vs_zic.py
"""
Simple test: PPOHandcrafted vs ZIC agents
This should be trivial - ZIC agents bid randomly, so any learning agent should dominate.
"""

# Test parameters
TOTAL_ROUNDS = 500
TRAINING_ROUNDS = 400
EVALUATION_ROUNDS = 100

CONFIG = {
    "experiment_name": "test_ppo_handcrafted_vs_zic",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": TRAINING_ROUNDS,
    
    # Market setup - wider price range for real trading opportunities
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,  # Changed to gametype with actual profit opportunities
    
    # Agent composition: 2 PPOHandcrafted + 4 ZIC buyers vs 6 ZIC sellers
    "buyers": [{"type": "ppo_handcrafted"}] * 2 + [{"type": "zic"}] * 4,
    "sellers": [{"type": "zic"}] * 6,
    
    # PPO parameters - keep it simple
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [128, 64],
        "lstm_hidden_size": 64,
        "lstm_num_layers": 1,
        
        # Training parameters
        "learning_rate": 3e-4,
        "batch_size": 512,
        "n_epochs": 5,
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
    "rng_seed_values": 42,
    "rng_seed_auction": 43,
    "rng_seed_rl": 44,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": False,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
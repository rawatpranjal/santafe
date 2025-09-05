# configs/ppo_convergence_test.py
"""
PPO Convergence Test with Winning Criteria
Train until performance stabilizes with rolling average tracking.
"""

CONFIG = {
    "experiment_name": "ppo_convergence_test",
    "experiment_dir": "experiments/active/ppo_convergence",
    "num_rounds": 2000,  # Max rounds (will stop early if converged)
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 2000,  # All training, no separate eval
    
    # Market setup - balanced for fair evaluation
    "num_buyers": 5,
    "num_sellers": 5,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # PPO vs diverse opponents
    "buyers": [
        {"type": "ppo_handcrafted"},  # 1 PPO
        {"type": "ppo_handcrafted"},  # 2 PPO  
        {"type": "zic"},              # 1 ZIC
        {"type": "zip"},              # 1 ZIP
        {"type": "zi"}                # 1 ZI
    ],
    "sellers": [
        {"type": "ppo_handcrafted"},  # 1 PPO
        {"type": "ppo_handcrafted"},  # 2 PPO
        {"type": "zic"},              # 1 ZIC
        {"type": "zip"},              # 1 ZIP
        {"type": "zi"}                # 1 ZI
    ],
    
    # Optimized PPO parameters
    "rl_params": {
        # Network architecture
        "nn_hidden_layers": [256, 128],
        
        # Proven hyperparameters
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.03,
        "max_grad_norm": 0.5,
        
        # Features for convergence
        "use_lr_annealing": True,
        "use_entropy_annealing": True,
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
    
    # Convergence criteria (for monitoring script)
    "convergence_criteria": {
        "window_size": 50,          # Rolling average window
        "min_rounds": 200,          # Minimum rounds before checking
        "stability_threshold": 0.05, # 5% variation = stable
        "profit_target": 10000,     # Minimum profit to be "winning"
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Logging
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
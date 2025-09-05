# configs/test_ppo_optimized_500.py
"""
Optimized PPO Handcrafted configuration for 500 rounds.
Uses all optimizations from "Implementation Matters in Deep Policy Gradients" (Engstrom et al.):
- Reward normalization with RunningMeanStd
- Learning rate annealing
- Entropy coefficient annealing 
- Target KL early stopping
- Value function clipping (already implemented)
- Orthogonal initialization (already implemented)
"""

CONFIG = {
    "experiment_name": "test_ppo_optimized_500",
    "experiment_dir": "experiments/ppo_optimized",
    "num_rounds": 500,  # Only 500 rounds needed with proper optimization
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 400,  # 400 training, 100 evaluation
    
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
    
    # PPO parameters - optimized feedforward network
    "rl_params": {
        # Network architecture - feedforward layers only
        "nn_hidden_layers": [256, 128],
        
        # Core PPO hyperparameters
        "learning_rate": 3e-4,  # Will be annealed to 0
        "batch_size": 2048,  # Large batch size for stable gradients
        "n_epochs": 10,  # Number of update epochs per period
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # GAE lambda
        "clip_epsilon": 0.2,  # PPO clip parameter
        "value_loss_coef": 0.5,  # Value loss coefficient
        "entropy_coef": 0.01,  # Initial entropy (will be annealed)
        "max_grad_norm": 0.5,  # Gradient clipping
        
        # Optimization features
        "use_lr_annealing": True,  # Anneal learning rate from 3e-4 to 0
        "use_entropy_annealing": True,  # Anneal entropy from 0.01 to 0
        "use_reward_scaling": True,  # Normalize rewards with running statistics
        "target_kl": 0.015,  # Stop epoch updates if KL exceeds this
        "norm_adv": True,  # Normalize advantages
        "clip_vloss": True,  # Clip value loss
        
        # Action space
        "num_price_actions": 31,  # More actions for finer control
        "price_range_pct": 0.15,  # Moderate range for exploration
        
        # Logging
        "log_level_rl": "INFO",
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
    "generate_eval_behavior_plots": True,  # Generate plots for evaluation periods
    "save_rl_model": True,
    "save_detailed_stats": True,
}
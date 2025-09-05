# configs/test_ppo_curriculum.py
"""
Curriculum learning for PPO: Start with ZIC opponents, gradually introduce ZIP.
First 200 rounds vs ZIC only, then mixed ZIC+ZIP, finally full ZIP competition.
"""

CONFIG = {
    "experiment_name": "test_ppo_curriculum",
    "experiment_dir": "experiments/ppo_curriculum",
    "num_rounds": 600,  # Extended for curriculum phases
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 500,  # 500 training, 100 evaluation
    
    # Market setup
    "num_buyers": 6,
    "num_sellers": 6,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Curriculum phases - implemented via custom training logic
    # Phase 1 (rounds 0-199): PPO vs ZIC only
    # Phase 2 (rounds 200-399): PPO vs mixed (2 ZIP, 1 ZIC each side)
    # Phase 3 (rounds 400-499): PPO vs ZIP only
    # Phase 4 (rounds 500-599): Evaluation
    
    # Agent composition for Phase 1 (will be modified programmatically)
    "buyers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    "sellers": [{"type": "ppo_handcrafted"}] * 3 + [{"type": "zic"}] * 3,
    
    # PPO parameters - adjusted for competitive learning
    "rl_params": {
        # Network architecture - feedforward layers only
        "nn_hidden_layers": [256, 128],
        
        # Core PPO hyperparameters - adjusted for exploration
        "learning_rate": 1e-4,  # Slower learning for stability
        "batch_size": 4096,  # Larger batch for stable gradients
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.03,  # Higher entropy for more exploration
        "max_grad_norm": 0.5,
        
        # Optimization features
        "use_lr_annealing": True,  # Anneal from 1e-4 to 0
        "use_entropy_annealing": False,  # Keep entropy high for exploration
        "use_reward_scaling": True,
        "target_kl": 0.02,  # Higher KL tolerance for exploration
        "norm_adv": True,
        "clip_vloss": True,
        
        # Action space - more granular for competitive markets
        "num_price_actions": 41,  # More price options
        "price_range_pct": 0.20,  # Wider exploration range
        
        # Exploration parameters
        "epsilon_greedy": 0.08,  # 8% random actions for exploration
        
        # Logging
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Curriculum learning parameters
    "curriculum_config": {
        "phase1_rounds": 200,  # PPO vs ZIC only
        "phase2_rounds": 200,  # PPO vs mixed (2 ZIP + 1 ZIC each side)
        "phase3_rounds": 100,  # PPO vs ZIP only
        "eval_rounds": 100,    # Evaluation against ZIP
    },
    
    # Seeds for reproducibility
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Logging and output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
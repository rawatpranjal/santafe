# templates/ppo_test.py
"""
Standard PPO testing configuration.
Use this for evaluating PPO performance against various opponents.
"""

CONFIG = {
    "experiment_name": "ppo_standard_test",
    "experiment_dir": "results/active/ppo_test",
    "num_rounds": 1000,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 800,
    
    # Market setup
    "num_buyers": 5,
    "num_sellers": 5,
    "num_tokens": 3,
    "min_price": 50,
    "max_price": 250,
    "gametype": 453,
    
    # Agent mix (customize as needed)
    "buyers": [
        {"type": "ppo_handcrafted"},  # 2 PPO
        {"type": "ppo_handcrafted"},
        {"type": "zic"},              # 2 ZIC
        {"type": "zic"},
        {"type": "zip"}               # 1 ZIP
    ],
    "sellers": [
        {"type": "ppo_handcrafted"},  # 2 PPO
        {"type": "ppo_handcrafted"},
        {"type": "zic"},              # 2 ZIC
        {"type": "zic"},
        {"type": "zip"}               # 1 ZIP
    ],
    
    # Standard PPO parameters (proven to work)
    "rl_params": {
        "nn_hidden_layers": [256, 128],
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.03,
        "max_grad_norm": 0.5,
        "use_lr_annealing": True,
        "use_reward_scaling": True,
        "target_kl": 0.02,
        "norm_adv": True,
        "clip_vloss": True,
        "num_price_actions": 41,
        "price_range_pct": 0.20,
        "epsilon_greedy": 0.05,
        "log_level_rl": "INFO",
        "log_training_stats": True,
    },
    
    # Seeds
    "rng_seed_values": 2024,
    "rng_seed_auction": 2025,
    "rng_seed_rl": 2026,
    
    # Logging
    "log_level": "INFO",
    "log_to_file": True,
    "save_rl_model": True,
    "save_detailed_stats": True,
}
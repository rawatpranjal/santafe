# templates/base_config.py
"""
Base configuration template for experiments.
Inherit from this for specific experiment types.
"""

class BaseConfig:
    """Base configuration with sensible defaults."""
    
    # Experiment metadata
    experiment_name = "base_experiment"
    experiment_dir = "results/active/base"
    
    # Simulation parameters
    num_rounds = 500
    num_periods = 3
    num_steps = 20
    num_training_rounds = 400  # 80% training, 20% eval
    
    # Market setup
    num_buyers = 5
    num_sellers = 5
    num_tokens = 3
    min_price = 50
    max_price = 250
    gametype = 453  # Default game type
    
    # Seeds for reproducibility
    rng_seed_values = 2024
    rng_seed_auction = 2025
    rng_seed_rl = 2026
    
    # Logging
    log_level = "INFO"
    log_to_file = True
    save_detailed_stats = True
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for main.py."""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }

# For compatibility with existing main.py
CONFIG = BaseConfig.to_dict()
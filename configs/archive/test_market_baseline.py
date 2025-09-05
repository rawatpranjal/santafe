# configs/test_market_baseline.py
"""
Baseline test: ZIC vs ZIP to verify market conditions allow profit.
This ensures we're not testing in degenerate markets.
"""

CONFIG = {
    "experiment_name": "test_market_baseline",
    "experiment_dir": "experiments/ppo_handcrafted_tests",
    "num_rounds": 50,
    "num_periods": 3,
    "num_steps": 20,
    "num_training_rounds": 0,  # No training, just baseline
    
    # Market setup - try different gametype for better surplus
    "num_buyers": 6,
    "num_sellers": 6, 
    "num_tokens": 3,
    "min_price": 1,
    "max_price": 200,
    "gametype": 108,  # Different gametype that should generate surplus
    
    # Agent composition: ZIC vs ZIP 
    "buyers": [{"type": "zic"}] * 3 + [{"type": "zip"}] * 3,
    "sellers": [{"type": "zic"}] * 3 + [{"type": "zip"}] * 3,
    
    # Seeds
    "rng_seed_values": 54321,
    "rng_seed_auction": 54322,
    
    # Output
    "log_level": "INFO",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "save_detailed_stats": True,
}
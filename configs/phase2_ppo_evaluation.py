# src_code/configs/phase2_ppo_evaluation.py
# Phase 2: PPO-LSTM Agent Evaluation
# Tests trained PPO-LSTM agents against all classical strategies
# Zero-shot generalization testing

import numpy as np

# --- Evaluation Parameters ---
TOTAL_ROUNDS = 1000  # Comprehensive evaluation
TRAINING_ROUNDS = 0  # No training, pure evaluation
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25
NUM_PERIODS = 3
NUM_TOKENS = 4

# Path to trained PPO model (from phase2_ppo_training.py)
TRAINED_MODEL_PATH = "experiments/phase2/training/phase2_ppo_vs_kaplan/models/final_model"

# --- Test Configurations ---

# Test 1: PPO vs each classical strategy individually
INDIVIDUAL_TESTS = []
classical_strategies = ["zic", "zip", "gd", "mgd", "el", "kaplan", "rg", "tt", "mu", "sk"]

for strategy in classical_strategies:
    config = {
        "experiment_name": f"phase2_eval_ppo_vs_{strategy}",
        "experiment_dir": "experiments/phase2/evaluation",
        "num_rounds": 500,  # 500 rounds per strategy
        "num_periods": NUM_PERIODS,
        "num_steps": STEPS_PER_PERIOD,
        "num_training_rounds": 0,  # No training
        "num_buyers": 6,
        "num_sellers": 6,
        "num_tokens": NUM_TOKENS,
        "min_price": 1,
        "max_price": 2000,
        "gametype": 6453,
        
        # 3 PPO buyers vs 3 strategy buyers, all strategy sellers
        "buyers": [{"type": "ppo_lstm"}] * 3 + [{"type": strategy}] * 3,
        "sellers": [{"type": strategy}] * 6,
        
        "rl_params": {
            "information_level": "extended",
            "num_price_actions": 21,
            "price_range_pct": 0.15,
            "log_level_rl": "WARNING",
        },
        "rng_seed_values": 5001 + classical_strategies.index(strategy),
        "rng_seed_auction": 5101 + classical_strategies.index(strategy),
        "rng_seed_rl": 5201 + classical_strategies.index(strategy),
        "log_level": "INFO",
        "log_level_rl": "WARNING",
        "log_to_file": True,
        "generate_per_round_plots": False,
        "generate_eval_behavior_plots": False,
        "num_eval_rounds_to_plot": 0,
        "save_rl_model": False,
        "load_rl_model_path": TRAINED_MODEL_PATH,
        "save_detailed_stats": True,
        "calculate_strategy_performance": True,
    }
    INDIVIDUAL_TESTS.append(config)

# Test 2: PPO in mixed market (all strategies present)
CONFIG_MIXED = {
    "experiment_name": "phase2_eval_ppo_mixed_market",
    "experiment_dir": "experiments/phase2/evaluation",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 0,
    "num_buyers": 11,  # 2 PPO + 9 classical (one of each)
    "num_sellers": 11,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 2 PPO buyers + one of each classical strategy
    "buyers": [{"type": "ppo_lstm"}] * 2 + [{"type": s} for s in classical_strategies[:9]],
    "sellers": [{"type": "ppo_lstm"}] * 2 + [{"type": s} for s in classical_strategies[:9]],
    
    "rl_params": {
        "information_level": "extended",
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "WARNING",
    },
    "rng_seed_values": 5301,
    "rng_seed_auction": 5302,
    "rng_seed_rl": 5303,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": False,
    "load_rl_model_path": TRAINED_MODEL_PATH,
    "save_detailed_stats": True,
    "calculate_strategy_rankings": True,
}

# Test 3: PPO dominance test (majority PPO agents)
CONFIG_DOMINANCE = {
    "experiment_name": "phase2_eval_ppo_dominance",
    "experiment_dir": "experiments/phase2/evaluation",
    "num_rounds": 500,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 0,
    "num_buyers": 8,
    "num_sellers": 8,
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,
    
    # 6 PPO vs 2 best classical (Kaplan/RG)
    "buyers": [{"type": "ppo_lstm"}] * 6 + [{"type": "kaplan"}] * 1 + [{"type": "rg"}] * 1,
    "sellers": [{"type": "ppo_lstm"}] * 6 + [{"type": "kaplan"}] * 1 + [{"type": "rg"}] * 1,
    
    "rl_params": {
        "information_level": "extended",
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "WARNING",
    },
    "rng_seed_values": 5401,
    "rng_seed_auction": 5402,
    "rng_seed_rl": 5403,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": False,
    "load_rl_model_path": TRAINED_MODEL_PATH,
    "save_detailed_stats": True,
}

# Test 4: Zero-shot generalization (new market conditions)
CONFIG_GENERALIZATION = {
    "experiment_name": "phase2_eval_ppo_generalization",
    "experiment_dir": "experiments/phase2/evaluation",
    "num_rounds": 500,
    "num_periods": 5,  # Different from training
    "num_steps": 50,    # Different from training
    "num_training_rounds": 0,
    "num_buyers": 15,   # Different market size
    "num_sellers": 15,
    "num_tokens": 8,    # Different token count
    "min_price": 1,
    "max_price": 5000,  # Different price range
    "gametype": 7891,   # Different value generation
    
    # Mix of PPO and various strategies
    "buyers": ([{"type": "ppo_lstm"}] * 5 + 
               [{"type": "zip"}] * 3 + 
               [{"type": "gd"}] * 3 + 
               [{"type": "kaplan"}] * 4),
    "sellers": ([{"type": "ppo_lstm"}] * 5 + 
                [{"type": "zip"}] * 3 + 
                [{"type": "gd"}] * 3 + 
                [{"type": "kaplan"}] * 4),
    
    "rl_params": {
        "information_level": "extended",
        "num_price_actions": 21,
        "price_range_pct": 0.15,
        "log_level_rl": "WARNING",
    },
    "rng_seed_values": 5501,
    "rng_seed_auction": 5502,
    "rng_seed_rl": 5503,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 10,
    "save_rl_model": False,
    "load_rl_model_path": TRAINED_MODEL_PATH,
    "save_detailed_stats": True,
}

# Main evaluation config
CONFIG = {
    "experiment_name": "phase2_ppo_comprehensive_eval",
    "experiment_dir": "experiments/phase2/evaluation",
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": 0,
    "evaluation_configs": INDIVIDUAL_TESTS + [CONFIG_MIXED, CONFIG_DOMINANCE, CONFIG_GENERALIZATION],
    
    # Default config for single run
    **CONFIG_MIXED,
}
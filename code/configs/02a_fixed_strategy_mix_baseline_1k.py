# code/configs/debug_rg_agent_margin100_30r.py
# DEBUGGING CONFIGURATION for Ringuette (RG) agent.
# - Uses the full mix of fixed strategies.
# - Overrides RG's profit_margin to 100 via init_args.
# - Sets log_level to DEBUG for detailed output.
# - Runs for only 30 rounds for faster analysis.

import numpy as np

# --- Timing and Phases ---
TOTAL_ROUNDS = 30   # Reduced rounds for debugging
TRAINING_ROUNDS = 0
EVALUATION_ROUNDS = TOTAL_ROUNDS

STEPS_PER_PERIOD = 25 # Standard SFI steps per period
NUM_PERIODS = 3     # Standard SFI periods per round
NUM_TOKENS = 4      # Standard tokens per agent per period

# List of all fixed strategies to include
strategy_types = ["zic", "zip", "gd", "el", "kaplan", "tt", "mu", "sk", "rg"]
num_strategies = len(strategy_types)

# --- Create Agent Specifications ---
# One of each type, but override RG's margin
agent_specs = []
for s_type in strategy_types:
    if s_type == "rg":
        # Override profit_margin for Ringuette agents
        agent_specs.append({"type": s_type, "init_args": {"profit_margin": 100}})
        print(f"DEBUG Config: Using profit_margin=100 for RG agents.") # Console message
    else:
        agent_specs.append({"type": s_type})


# --- CONFIG Dictionary ---
CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": "debug_rg_agent_margin100_30r", # Descriptive name
    "experiment_dir": "experiments",

    # --- Auction Settings ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": NUM_PERIODS,
    "num_steps": STEPS_PER_PERIOD,
    "num_training_rounds": TRAINING_ROUNDS,
    "num_buyers": num_strategies,    # 9 Buyers
    "num_sellers": num_strategies,   # 9 Sellers
    "num_tokens": NUM_TOKENS,
    "min_price": 1,
    "max_price": 2000,
    "gametype": 6453,   # Standard SFI gametype

    # --- Trader Configuration ---
    "buyers": agent_specs, # Use the list with modified RG spec
    "sellers": agent_specs, # Use the list with modified RG spec

    # --- PPO/RL Hyperparameters (Not used) ---
    "rl_params": {},

    # --- RNG Seeds, Logging, Plotting, Saving ---
    "rng_seed_values": 2027, # Keep consistent seeds if desired for comparison
    "rng_seed_auction": 414,
    "rng_seed_rl": 525,
    "log_level": "DEBUG",    # <--- SET TO DEBUG for detailed logs
    "log_level_rl": "WARNING", # Keep RL logs quiet
    "log_to_file": True,
    "generate_per_round_plots": False, # Turn off for speed
    "generate_eval_behavior_plots": False,
    "num_eval_rounds_to_plot": 0,
    "save_rl_model": False,
    "load_rl_model_path": None,
}

print(f"--- Loaded DEBUG config: {CONFIG['experiment_name']} ---")
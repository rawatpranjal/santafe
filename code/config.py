# config.py
import random

# --- (Keep timing, RL, participant counts etc. as before) ---
TOTAL_ROUNDS = 5000 # Or 50000, etc.
TRAINING_FRACTION = 0.95
EVALUATION_ROUNDS = int(TOTAL_ROUNDS * (1.0 - TRAINING_FRACTION))
TRAINING_ROUNDS = TOTAL_ROUNDS - EVALUATION_ROUNDS
STEPS_PER_ROUND = 4 * 25
TOTAL_TRAINING_STEPS = TRAINING_ROUNDS * STEPS_PER_ROUND

CONFIG = {
    # --- Experiment Identification ---
    "experiment_name": f"006_dqn_vs_zic_kaplan_{TOTAL_ROUNDS}r", # New name
    "experiment_dir": "experiments",

    # --- (Auction Timing, RL Training, Participant counts etc.) ---
    "num_rounds": TOTAL_ROUNDS,
    "num_periods": 4,
    "num_steps": 25,
    "num_training_rounds": TRAINING_ROUNDS,
    "rl_agent_type": "dqn_pricegrid", # Still matches the DQN type below
    "num_buyers": 3,
    "num_sellers": 3,
    "num_tokens": 4,
    "min_price": 1,
    "max_price": 200,
    "gametype": 1236,

    # --- Trader Configuration (DQN vs ZIC vs Kaplan) ---
    "buyers": [
        {"type": "dqn_pricegrid", # B0 is still DQN
         "init_args": {
             "num_price_actions": 21,
             "price_range_pct": 0.15
             }
        },
        {"type": "zic"},         # B1 is ZIC
        {"type": "zic"},         # B1 is ZIC
        {"type": "zic"},         # B1 is ZIC
        {"type": "kaplan"},      # B2 is now Kaplan
    ],
    "sellers": [
        {"type": "zic"},         # S0 is ZIC
        {"type": "zic"},         # B1 is ZIC
        {"type": "zic"},         # B1 is ZIC
        {"type": "kaplan"},      # S1 is now Kaplan
        {"type": "dqn_pricegrid", # B0 is still DQN
         "init_args": {
             "num_price_actions": 21,
             "price_range_pct": 0.15
             }
        },
    ],

    # --- (DQN Hyperparameters remain the same) ---
     "rl_params": {
        "buffer_size": 150_000,
        "batch_size": 128,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": int(TOTAL_TRAINING_STEPS * 0.80),
        "learning_rate": 0.0005,
        "target_update_freq": 1000,
        "nn_hidden_layers": [128, 64],
        "optimizer": "Adam",
        "gradient_clip_value": 1.0,
    },

    # --- (RNG Seeds, Logging, Plotting, Saving remain the same) ---
    "rng_seed_values": 2026, # Update seeds for new mix
    "rng_seed_auction": 409,
    "rng_seed_rl": 520,
    "log_level": "INFO",
    "log_level_rl": "WARNING",
    "log_to_file": True,
    "generate_per_round_plots": False,
    "generate_eval_behavior_plots": True,
    "num_eval_rounds_to_plot": 5,
    "save_rl_model": True,
    "load_rl_model_path": None,
}

# Print calculated values for verification
print(f"Total Rounds: {TOTAL_ROUNDS}")
print(f"Calculated Training Rounds: {TRAINING_ROUNDS}")
print(f"Calculated Evaluation Rounds: {EVALUATION_ROUNDS}")
print(f"Calculated Total Training Steps: {TOTAL_TRAINING_STEPS}")
print(f"Calculated Epsilon Decay Steps: {CONFIG['rl_params']['epsilon_decay_steps']}")
#!/usr/bin/env python3
"""Quick test of PPO handcrafted performance after extended training.

NOTE: This is a v1.0 legacy test file. Disabled in v2.0 until PPO agents are reimplemented.
All code in this file is wrapped in `if False` to prevent execution during test collection.
"""

import pytest

# This file contains no actual test functions - it's a standalone script
# Skip marker prevents pytest from complaining
pytestmark = pytest.mark.skip(reason="v1.0 legacy test - not a proper test file")


def test_ppo_placeholder() -> None:
    """Placeholder test to satisfy pytest structure."""
    pytest.skip("PPO tests will be reimplemented in Phase 2")


# The original file content is disabled below
# It will be removed/rewritten when Phase 2 (Legacy Agents) begins

if False:  # DISABLED v1.0 legacy code
    import sys
    import os
    sys.path.append('src_code')

    from auction import Auction
    import numpy as np

    config = {
        "experiment_name": "quick_ppo_test",
        "num_rounds": 200,
        "num_periods": 3,
        "steps_per_period": 20,
        "train_rounds": 150,

        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 3,
        "min_price": 50,
        "max_price": 250,
        "gametype": 453,

        "buyers": [{"type": "ppo_handcrafted"}] + [{"type": "zic"}] * 3,
        "sellers": [{"type": "zic"}] * 4,

        "rl_params": {
            "nn_hidden_layers": [256, 128],
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "learning_rate": 3e-4,
            "batch_size": 1024,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.02,
            "max_grad_norm": 0.5,
            "num_price_actions": 101,
            "price_range_pct": 1.0,
            "use_reward_scaling": False,
            "log_level_rl": "WARNING",
            "log_training_stats": False,
        },

        "rng_seed_values": 42,
        "rng_seed_auction": 43,
        "rng_seed_rl": 44,
        "log_level": "WARNING",
        "log_to_file": False,
        "generate_per_round_plots": False,
        "generate_eval_behavior_plots": False,
        "save_rl_model": False,
        "save_detailed_stats": False,
    }

    # Run auction
    auction = Auction(config)
    auction.run()

    # Get results
    eval_start = config["train_rounds"]
    buyer_profits = {}

    for buyer in auction.buyers:
        total_profit = 0
        total_trades = 0
        for r in range(eval_start, config["num_rounds"]):
            round_data = auction.round_history[r]
            if buyer.name in round_data.get("profits", {}):
                total_profit += round_data["profits"][buyer.name]
            if buyer.name in round_data.get("trades", {}):
                total_trades += round_data["trades"][buyer.name]

        if total_trades > 0:
            buyer_profits[buyer.name] = total_profit / total_trades
        else:
            buyer_profits[buyer.name] = 0

    print("\n=== Quick PPO Test Results ===")
    print("Mean profit per trade by buyer:")
    for name, profit in sorted(buyer_profits.items(), key=lambda x: x[0]):
        agent_type = "PPO" if "B0" in name else "ZIC"
        print(f"  {name} ({agent_type}): {profit:.2f}")

    # Rank buyers
    ranked = sorted(buyer_profits.items(), key=lambda x: x[1], reverse=True)
    print("\nBuyer rankings:")
    for i, (name, profit) in enumerate(ranked, 1):
        agent_type = "PPO" if "B0" in name else "ZIC"
        print(f"  {i}. {name} ({agent_type}): {profit:.2f}")

    # PPO performance vs ZIC average
    ppo_profit = buyer_profits.get("B0", 0)
    zic_profits = [p for n, p in buyer_profits.items() if n != "B0"]
    zic_avg = np.mean(zic_profits) if zic_profits else 0

    print(f"\nPPO vs ZIC comparison:")
    print(f"  PPO profit/trade: {ppo_profit:.2f}")
    print(f"  ZIC avg profit/trade: {zic_avg:.2f}")
    if zic_avg > 0:
        print(f"  PPO captures {100 * ppo_profit / zic_avg:.1f}% of ZIC performance")

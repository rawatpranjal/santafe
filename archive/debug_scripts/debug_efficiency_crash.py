#!/usr/bin/env python3
"""
Debug script to understand 0% efficiency in pure ZIC tournament.
"""

import sys
sys.path.append('.')

from omegaconf import OmegaConf
from engine.tournament import Tournament
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pure ZIC configuration matching tournament
config = OmegaConf.create({
    "experiment": {
        "name": "debug_zic_crash",
        "output_dir": "results/debug_zic_crash",
        "num_rounds": 15,  # Run just first 15 rounds to capture crash
        "rng_seed_auction": 42,
        "rng_seed_values": 123,
        "log_level": "INFO"
    },
    "market": {
        "num_tokens": 4,
        "num_periods": 2,  # Just 2 periods for faster debugging
        "num_steps": 100,
        "min_price": 1,
        "max_price": 1000,
        "gametype": 6453  # Same as tournament
    },
    "agents": {
        "buyer_types": ['ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC'],
        "seller_types": ['ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC', 'ZIC']
    }
})

# Enable diagnostic logging for all rounds
import engine.tournament as tourn_module

# Run tournament
tournament = Tournament(config)
results = tournament.run()

# Analyze results
print("\n" + "="*60)
print("EFFICIENCY BY ROUND")
print("="*60)

for r in range(1, 16):
    round_data = results[results['round'] == r]
    if len(round_data) > 0:
        avg_eff = round_data['efficiency'].mean()
        agent_1_data = round_data[round_data['agent_id'] == 1]
        trades = agent_1_data['num_trades'].sum() if len(agent_1_data) > 0 else 0

        status = "✅" if avg_eff > 80 else ("⚠️" if avg_eff > 50 else "❌")
        print(f"Round {r:2d}: {avg_eff:6.2f}% efficiency, {trades} trades {status}")

print("="*60)

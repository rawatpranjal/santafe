#!/usr/bin/env python3
"""
Quick test to verify the state management fix for ZIC traders.
Tests a simple ZIC vs ZIC market across multiple rounds.
"""

import sys
sys.path.append('.')

from omegaconf import OmegaConf
from engine.tournament import Tournament
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple ZIC vs ZIC configuration
config = OmegaConf.create({
    "experiment": {
        "name": "test_zic_fix",
        "output_dir": "results/test_zic_fix",
        "num_rounds": 5,
        "rng_seed_auction": 42,
        "rng_seed_values": 123
    },
    "market": {
        "num_tokens": 4,
        "num_periods": 2,  # Quick test with 2 periods per round
        "num_steps": 100,
        "min_price": 0,
        "max_price": 100,
        "gametype": 64213  # BASE configuration
    },
    "agents": {
        "buyer_types": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "seller_types": ["ZIC", "ZIC", "ZIC", "ZIC"]
    }
})

# Run tournament
tournament = Tournament(config)
results = tournament.run()

# Check results
print("\n" + "="*60)
print("ZIC STATE MANAGEMENT FIX TEST RESULTS")
print("="*60)

# Group by round and check efficiency
for round_num in range(1, 6):
    round_data = results[results['round'] == round_num]
    if len(round_data) > 0:
        avg_efficiency = round_data['efficiency'].mean()
        print(f"Round {round_num}: Average Efficiency = {avg_efficiency:.2f}%")

        # Check if efficiency is reasonable (should be ~97-98% for ZIC)
        if avg_efficiency < 90:
            print(f"  WARNING: Efficiency too low! Expected ~97-98% for ZIC")
        elif avg_efficiency > 100:
            print(f"  ERROR: Efficiency > 100%! Calculation error")
        else:
            print(f"  ✓ Efficiency looks reasonable")

print("\n" + "="*60)
print("OVERALL RESULTS:")
print(f"Mean efficiency: {results['efficiency'].mean():.2f}%")
print(f"Std deviation: {results['efficiency'].std():.2f}%")

# Success criteria
if results['efficiency'].mean() > 95 and results['efficiency'].mean() < 100:
    print("\n✅ TEST PASSED: State management fix appears to be working!")
else:
    print("\n❌ TEST FAILED: Efficiency outside expected range for ZIC")

print("="*60)
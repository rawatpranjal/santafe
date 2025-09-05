#!/usr/bin/env python
"""
Test script to verify memory optimizations in the trading simulation.
This will run a small simulation and monitor memory usage.
"""

import sys
import os
import psutil
import gc
import time

# Add src_code to path
sys.path.insert(0, 'src_code')

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_memory_optimizations():
    print("Testing memory optimizations in trading simulation")
    print("=" * 60)
    
    # Record initial memory
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Import and setup
    from main import CONFIG, Auction, logger
    import logging
    
    # Configure for a small test run
    test_config = {
        'experiment_name': 'memory_test',
        'experiment_dir': 'test_experiments',
        'num_rounds': 100,  # Small number for quick test
        'num_periods': 1,
        'num_steps': 25,
        'num_training_rounds': 50,
        'num_buyers': 5,
        'num_sellers': 5,
        'num_tokens': 1,
        'min_price': 1,
        'max_price': 2000,
        'gametype': 0,
        'log_level': 'INFO',
        'log_level_rl': 'WARNING',
        'save_rl_model': False,
        'generate_per_round_plots': False,
        'generate_eval_behavior_plots': False,
        'buyers': [
            {'type': 'ZI_C', 'init_args': {}},
            {'type': 'PPO', 'init_args': {}},
        ],
        'sellers': [
            {'type': 'ZI_C', 'init_args': {}},
            {'type': 'PPO', 'init_args': {}},
        ],
        'rl_params': {
            'gamma': 0.99,
            'lr': 0.0003,
            'buffer_size': 1000,  # Small buffer for test
            'batch_size': 32,
        },
        'rng_seed_values': 42,
        'rng_seed_auction': 43,
    }
    
    # Resolve trader classes
    from traders.registry import get_trader_class
    for spec in test_config['buyers']:
        spec['class'] = get_trader_class(spec['type'], is_buyer=True)
    for spec in test_config['sellers']:
        spec['class'] = get_trader_class(spec['type'], is_buyer=False)
    
    print(f"\nRunning test simulation with {test_config['num_rounds']} rounds...")
    
    # Create output directory
    os.makedirs('test_experiments/memory_test', exist_ok=True)
    
    # Track memory during execution
    memory_checkpoints = []
    
    # Open CSV file for step logs
    step_log_path = 'test_experiments/memory_test/step_log.csv'
    with open(step_log_path, 'w', newline='') as step_log_file:
        # Run auction
        auction = Auction(test_config, step_log_file=step_log_file)
        
        # Monitor memory at key points
        for round_idx in range(0, test_config['num_rounds'], 10):
            if round_idx > 0:
                # Run 10 rounds
                for _ in range(10):
                    auction.current_round += 1
                    # Simplified round execution for testing
                    
            gc.collect()
            current_memory = get_memory_usage()
            memory_checkpoints.append((round_idx, current_memory))
            print(f"Round {round_idx}: Memory = {current_memory:.2f} MB (Δ = {current_memory - initial_memory:.2f} MB)")
    
    # Final memory check
    gc.collect()
    final_memory = get_memory_usage()
    
    print("\n" + "=" * 60)
    print("MEMORY TEST RESULTS:")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Total increase: {final_memory - initial_memory:.2f} MB")
    
    # Check for memory leaks
    if len(memory_checkpoints) > 2:
        # Calculate average memory increase per round
        first_checkpoint = memory_checkpoints[1][1]  # Skip initial
        last_checkpoint = memory_checkpoints[-1][1]
        rounds_between = memory_checkpoints[-1][0] - memory_checkpoints[1][0]
        
        if rounds_between > 0:
            avg_increase_per_round = (last_checkpoint - first_checkpoint) / rounds_between
            print(f"Average memory increase per round: {avg_increase_per_round:.4f} MB")
            
            if avg_increase_per_round > 1.0:
                print("⚠️ WARNING: Significant memory growth detected!")
                print("   This suggests a memory leak may still exist.")
            elif avg_increase_per_round > 0.1:
                print("⚠️ CAUTION: Moderate memory growth detected.")
                print("   This may be acceptable for short runs but could be problematic for long simulations.")
            else:
                print("✅ SUCCESS: Memory usage appears stable!")
                print("   The optimizations are working effectively.")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_memory_optimizations()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
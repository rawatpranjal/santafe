#!/usr/bin/env python3
"""
Monitor PPO convergence and stop when stable.
Tracks rolling average profit and checks for convergence.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path

def check_convergence(experiment_dir, window_size=50, min_rounds=200, 
                     stability_threshold=0.05, profit_target=10000):
    """Check if PPO has converged based on rolling average stability."""
    
    step_log = Path(experiment_dir) / "step_log_all.csv"
    
    if not step_log.exists():
        return False, 0, 0, "No log file yet"
    
    try:
        df = pd.read_csv(step_log)
        
        # Filter for PPO agents
        if 'trader_type' in df.columns:
            ppo_df = df[df['trader_type'] == 'ppo_handcrafted']
        else:
            return False, 0, 0, "No trader_type column"
        
        if 'round' not in ppo_df.columns or 'profit' not in ppo_df.columns:
            return False, 0, 0, "Missing required columns"
        
        # Group by round
        round_profits = ppo_df.groupby('round')['profit'].sum().reset_index()
        n_rounds = len(round_profits)
        
        if n_rounds < min_rounds:
            return False, n_rounds, 0, f"Only {n_rounds}/{min_rounds} rounds"
        
        # Calculate rolling average
        round_profits['rolling_avg'] = round_profits['profit'].rolling(
            window=window_size, min_periods=1).mean()
        
        # Get recent data
        recent = round_profits.tail(window_size)
        current_avg = recent['rolling_avg'].mean()
        
        # Check if profitable enough
        if current_avg < profit_target:
            return False, n_rounds, current_avg, f"Below target: {current_avg:.0f} < {profit_target}"
        
        # Check stability (coefficient of variation)
        std = recent['rolling_avg'].std()
        cv = std / current_avg if current_avg > 0 else float('inf')
        
        if cv < stability_threshold:
            return True, n_rounds, current_avg, f"CONVERGED! Avg: {current_avg:.0f}, CV: {cv:.3f}"
        else:
            return False, n_rounds, current_avg, f"Still learning: Avg: {current_avg:.0f}, CV: {cv:.3f}"
            
    except Exception as e:
        return False, 0, 0, f"Error: {e}"

def monitor_experiment(config_file="src_code/configs/ppo_convergence_test.py"):
    """Monitor experiment and stop when converged."""
    
    # Import config to get parameters
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.CONFIG
    
    experiment_dir = config["experiment_dir"]
    criteria = config.get("convergence_criteria", {})
    
    window = criteria.get("window_size", 50)
    min_rounds = criteria.get("min_rounds", 200)
    threshold = criteria.get("stability_threshold", 0.05)
    target = criteria.get("profit_target", 10000)
    
    print(f"Monitoring {experiment_dir}")
    print(f"Convergence criteria:")
    print(f"  - Window: {window} rounds")
    print(f"  - Min rounds: {min_rounds}")
    print(f"  - Stability: {threshold:.1%} CV")
    print(f"  - Profit target: {target}")
    print()
    
    check_interval = 10  # Check every 10 seconds
    
    while True:
        converged, rounds, avg_profit, status = check_convergence(
            experiment_dir, window, min_rounds, threshold, target)
        
        print(f"Round {rounds:4d}: {status}")
        
        if converged:
            print("\n✓ CONVERGENCE ACHIEVED!")
            print(f"Final average profit: {avg_profit:.0f}")
            print(f"Completed in {rounds} rounds")
            
            # Archive to wins folder
            wins_dir = Path("experiments/wins") / f"converged_{rounds}rounds_{avg_profit:.0f}profit"
            print(f"\nArchiving to: {wins_dir}")
            
            # Could copy files here if needed
            
            return True
        
        time.sleep(check_interval)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "src_code/configs/ppo_convergence_test.py"
    
    monitor_experiment(config_file)
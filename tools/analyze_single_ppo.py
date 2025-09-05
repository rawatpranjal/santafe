#!/usr/bin/env python3
"""
Single PPO Performance Analysis
Analyze how well a single PPO agent pair performs against mixed opponents.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def analyze_single_ppo_performance(experiment_dir="experiments/single_ppo_vs_mixed/single_ppo_vs_mixed"):
    """Analyze single PPO agent performance against mixed opponents."""
    
    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Look for step logs
    step_log_path = os.path.join(experiment_dir, "step_log_all.csv")
    if not os.path.exists(step_log_path):
        print(f"Step log not found: {step_log_path}")
        return
    
    print(f"Analyzing PPO performance from: {step_log_path}")
    
    try:
        # Read step logs
        df = pd.read_csv(step_log_path)
        print(f"Loaded {len(df)} trading steps from logs")
        
        # Check what columns we have
        print(f"Available columns: {list(df.columns)}")
        
        # Group by round to analyze round-level performance
        if 'round' in df.columns:
            round_stats = df.groupby('round').agg({
                'profit': ['sum', 'mean', 'std'],
                'efficiency': 'mean'
            }).reset_index()
            
            # Flatten column names
            round_stats.columns = ['round', 'total_profit', 'avg_profit', 'profit_std', 'efficiency']
            
            # Filter for PPO agents if trader_type column exists
            if 'trader_type' in df.columns:
                ppo_data = df[df['trader_type'] == 'ppo_handcrafted']
                if len(ppo_data) > 0:
                    ppo_round_stats = ppo_data.groupby('round').agg({
                        'profit': ['sum', 'mean', 'count']
                    }).reset_index()
                    ppo_round_stats.columns = ['round', 'ppo_total_profit', 'ppo_avg_profit', 'ppo_agent_count']
                    
                    print("\n=== PPO PERFORMANCE ANALYSIS ===")
                    print(f"PPO agents tracked: {ppo_round_stats['ppo_agent_count'].iloc[0] if len(ppo_round_stats) > 0 else 'Unknown'}")
                    
                    if len(ppo_round_stats) >= 10:
                        # Calculate rolling averages
                        ppo_round_stats['ppo_profit_roll20'] = ppo_round_stats['ppo_total_profit'].rolling(20, min_periods=1).mean()
                        
                        # Latest performance
                        latest_rounds = ppo_round_stats.tail(20)
                        avg_recent_profit = latest_rounds['ppo_total_profit'].mean()
                        
                        print(f"Recent 20-round average PPO profit: {avg_recent_profit:.1f}")
                        print(f"Best single round PPO profit: {ppo_round_stats['ppo_total_profit'].max():.1f}")
                        print(f"Worst single round PPO profit: {ppo_round_stats['ppo_total_profit'].min():.1f}")
                        
                        # Trend analysis
                        early_avg = ppo_round_stats.head(50)['ppo_total_profit'].mean()
                        recent_avg = ppo_round_stats.tail(50)['ppo_total_profit'].mean()
                        improvement = ((recent_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
                        
                        print(f"Early performance (first 50 rounds): {early_avg:.1f}")
                        print(f"Recent performance (last 50 rounds): {recent_avg:.1f}")
                        print(f"Learning improvement: {improvement:+.1f}%")
            
        # Overall statistics
        print(f"\n=== OVERALL MARKET STATISTICS ===")
        if 'efficiency' in df.columns:
            print(f"Average market efficiency: {df['efficiency'].mean():.3f}")
            print(f"Efficiency std dev: {df['efficiency'].std():.3f}")
        
        if 'profit' in df.columns:
            total_profit = df['profit'].sum()
            print(f"Total market profit: {total_profit:.1f}")
            print(f"Average step profit: {df['profit'].mean():.3f}")
        
        # Agent type breakdown if available
        if 'trader_type' in df.columns:
            agent_performance = df.groupby('trader_type').agg({
                'profit': ['sum', 'mean', 'count']
            })
            agent_performance.columns = ['total_profit', 'avg_profit', 'step_count']
            
            print(f"\n=== PERFORMANCE BY AGENT TYPE ===")
            for agent_type in agent_performance.index:
                total = agent_performance.loc[agent_type, 'total_profit']
                avg = agent_performance.loc[agent_type, 'avg_profit']
                count = agent_performance.loc[agent_type, 'step_count']
                print(f"{agent_type:15s}: Total={total:8.1f}, Avg={avg:6.3f}, Steps={count:6d}")
    
    except Exception as e:
        print(f"Error analyzing performance: {e}")
        return
    
    print(f"\nAnalysis complete. Full logs available at: {experiment_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        experiment_dir = "experiments/single_ppo_vs_mixed/single_ppo_vs_mixed"
    
    analyze_single_ppo_performance(experiment_dir)
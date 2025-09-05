#!/usr/bin/env python3
"""
Track PPO performance over time during extended training.
Analyzes round_log_all.csv to show learning curves.
"""

import pandas as pd
import numpy as np
import ast
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_ppo_performance(csv_path, window=50):
    """Analyze PPO performance from round log CSV."""
    
    df = pd.read_csv(csv_path)
    
    # Initialize tracking lists
    rounds = []
    ppo_avg_profits = []
    zic_avg_profits = []
    ppo_profit_ratios = []
    market_efficiencies = []
    
    # Process each round
    for _, row in df.iterrows():
        round_num = row['round']
        bot_details = ast.literal_eval(row['bot_details'])
        
        # Extract profits by strategy
        ppo_profits = [b['profit'] for b in bot_details if b['strategy'] == 'ppo_handcrafted']
        zic_profits = [b['profit'] for b in bot_details if b['strategy'] == 'zic']
        
        if ppo_profits and zic_profits:
            rounds.append(round_num)
            ppo_avg = np.mean(ppo_profits) if ppo_profits else 0
            zic_avg = np.mean(zic_profits) if zic_profits else 0
            
            ppo_avg_profits.append(ppo_avg)
            zic_avg_profits.append(zic_avg)
            
            # Calculate profit ratio (PPO / ZIC)
            ratio = ppo_avg / zic_avg if zic_avg > 0 else 0
            ppo_profit_ratios.append(ratio)
            
            market_efficiencies.append(row['market_efficiency'])
    
    # Create DataFrame for analysis
    results = pd.DataFrame({
        'round': rounds,
        'ppo_avg_profit': ppo_avg_profits,
        'zic_avg_profit': zic_avg_profits,
        'profit_ratio': ppo_profit_ratios,
        'market_efficiency': market_efficiencies
    })
    
    # Calculate rolling averages
    if window > 0 and len(results) > window:
        results['ppo_rolling_avg'] = results['ppo_avg_profit'].rolling(window=window, min_periods=1).mean()
        results['zic_rolling_avg'] = results['zic_avg_profit'].rolling(window=window, min_periods=1).mean()
        results['ratio_rolling_avg'] = results['profit_ratio'].rolling(window=window, min_periods=1).mean()
    else:
        results['ppo_rolling_avg'] = results['ppo_avg_profit']
        results['zic_rolling_avg'] = results['zic_avg_profit']
        results['ratio_rolling_avg'] = results['profit_ratio']
    
    return results

def plot_performance(results, output_dir):
    """Create performance plots."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Absolute profits
    ax = axes[0]
    ax.plot(results['round'], results['ppo_avg_profit'], alpha=0.3, label='PPO (raw)', color='blue')
    ax.plot(results['round'], results['ppo_rolling_avg'], label='PPO (rolling avg)', color='blue', linewidth=2)
    ax.plot(results['round'], results['zic_avg_profit'], alpha=0.3, label='ZIC (raw)', color='orange')
    ax.plot(results['round'], results['zic_rolling_avg'], label='ZIC (rolling avg)', color='orange', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Average Profit')
    ax.set_title('PPO vs ZIC Profit Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Profit ratio (PPO / ZIC)
    ax = axes[1]
    ax.plot(results['round'], results['profit_ratio'], alpha=0.3, color='green')
    ax.plot(results['round'], results['ratio_rolling_avg'], label='PPO/ZIC Ratio', color='green', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Equal performance')
    ax.set_xlabel('Round')
    ax.set_ylabel('Profit Ratio (PPO / ZIC)')
    ax.set_title('Relative Performance: PPO vs ZIC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Market efficiency
    ax = axes[2]
    ax.plot(results['round'], results['market_efficiency'], alpha=0.5, color='purple')
    window = min(50, len(results) // 10)
    if window > 1:
        eff_rolling = results['market_efficiency'].rolling(window=window, min_periods=1).mean()
        ax.plot(results['round'], eff_rolling, label=f'{window}-round avg', color='purple', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Market Efficiency')
    ax.set_title('Market Efficiency Over Time')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'ppo_performance_tracking.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    
    return fig

def print_summary(results):
    """Print performance summary."""
    
    print("\n" + "="*60)
    print("PPO PERFORMANCE SUMMARY")
    print("="*60)
    
    # Split into quarters for analysis
    n_rounds = len(results)
    q1 = n_rounds // 4
    q2 = n_rounds // 2
    q3 = 3 * n_rounds // 4
    
    print("\n--- Average Profits ---")
    print(f"First {q1} rounds:")
    print(f"  PPO: {results['ppo_avg_profit'][:q1].mean():.2f}")
    print(f"  ZIC: {results['zic_avg_profit'][:q1].mean():.2f}")
    print(f"  Ratio: {results['profit_ratio'][:q1].mean():.3f}")
    
    print(f"\nRounds {q1}-{q2}:")
    print(f"  PPO: {results['ppo_avg_profit'][q1:q2].mean():.2f}")
    print(f"  ZIC: {results['zic_avg_profit'][q1:q2].mean():.2f}")
    print(f"  Ratio: {results['profit_ratio'][q1:q2].mean():.3f}")
    
    print(f"\nRounds {q2}-{q3}:")
    print(f"  PPO: {results['ppo_avg_profit'][q2:q3].mean():.2f}")
    print(f"  ZIC: {results['zic_avg_profit'][q2:q3].mean():.2f}")
    print(f"  Ratio: {results['profit_ratio'][q2:q3].mean():.3f}")
    
    print(f"\nLast {n_rounds-q3} rounds:")
    print(f"  PPO: {results['ppo_avg_profit'][q3:].mean():.2f}")
    print(f"  ZIC: {results['zic_avg_profit'][q3:].mean():.2f}")
    print(f"  Ratio: {results['profit_ratio'][q3:].mean():.3f}")
    
    # Check for improvement
    early_ratio = results['profit_ratio'][:100].mean() if len(results) > 100 else results['profit_ratio'][:q1].mean()
    late_ratio = results['profit_ratio'][-100:].mean() if len(results) > 100 else results['profit_ratio'][q3:].mean()
    
    print(f"\n--- Learning Progress ---")
    print(f"Early ratio (first 100): {early_ratio:.3f}")
    print(f"Late ratio (last 100): {late_ratio:.3f}")
    print(f"Improvement: {(late_ratio - early_ratio):.3f} ({(late_ratio/early_ratio - 1)*100:.1f}%)")
    
    if late_ratio > 1.0:
        print("\n✅ SUCCESS: PPO agents are outperforming ZIC!")
    elif late_ratio > 0.8:
        print("\n⚠️ PARTIAL: PPO agents are competitive but not dominant")
    else:
        print("\n❌ UNDERPERFORMING: PPO agents need more tuning")
    
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_ppo_performance.py <path_to_round_log.csv>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Analyze performance
    results = analyze_ppo_performance(csv_path)
    
    # Print summary
    print_summary(results)
    
    # Create plots
    output_dir = csv_path.parent
    plot_performance(results, output_dir)
    
    # Save processed data
    processed_path = output_dir / 'ppo_performance_processed.csv'
    results.to_csv(processed_path, index=False)
    print(f"\nSaved processed data to {processed_path}")
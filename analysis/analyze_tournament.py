#!/usr/bin/env python3
"""
Tournament Analysis for PPOHandcraftedTrader
Generates quantitative validation tables proving agent effectiveness.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from scipy import stats
import json


def analyze_strategy_tournament(df_rounds, eval_start_round):
    """
    Analyze tournament rankings by strategy.
    Returns table with MeanProfit, MeanRank, and SuccessRate.
    """
    eval_df = df_rounds[df_rounds['round'] >= eval_start_round].copy()
    
    # Collect all agent performance data
    all_agents = []
    for _, row in eval_df.iterrows():
        bot_details = ast.literal_eval(row['bot_details'])
        for bot in bot_details:
            agent_data = {
                'round': row['round'],
                'name': bot['name'],
                'role': bot['role'],
                'strategy': bot['strategy'],
                'profit': bot['profit'],
                'trades': bot['trades'],
            }
            all_agents.append(agent_data)
    
    agents_df = pd.DataFrame(all_agents)
    
    # Calculate ranks within each round
    agents_df['rank'] = agents_df.groupby('round')['profit'].rank(
        method='average', ascending=False
    )
    
    # Group by strategy
    strategy_stats = []
    for strategy in agents_df['strategy'].unique():
        strat_df = agents_df[agents_df['strategy'] == strategy]
        
        # Calculate statistics
        mean_profit = strat_df['profit'].mean()
        std_profit = strat_df['profit'].std()
        mean_rank = strat_df['rank'].mean()
        std_rank = strat_df['rank'].std()
        
        # Success rate (% of rounds with positive profit)
        success_rate = (strat_df['profit'] > 0).mean()
        
        # Profit ratio (mean profit / mean cost)
        profit_ratio = strat_df['profit'].sum() / max(strat_df['trades'].sum(), 1)
        
        # Trade execution rate
        trade_rate = strat_df['trades'].mean()
        
        strategy_stats.append({
            'Strategy': strategy,
            'Mean Profit': f"{mean_profit:.2f}",
            '(Std Dev)': f"({std_profit:.2f})",
            'Mean Rank': f"{mean_rank:.2f}",
            '(Rank Std)': f"({std_rank:.2f})",
            'Success Rate': f"{success_rate:.1%}",
            'Trade Rate': f"{trade_rate:.2f}",
            'Agent-Rounds': len(strat_df)
        })
    
    # Sort by mean profit
    strategy_stats = sorted(strategy_stats, 
                           key=lambda x: float(x['Mean Profit']), 
                           reverse=True)
    
    return pd.DataFrame(strategy_stats)


def analyze_market_efficiency(df_rounds, eval_start_round):
    """
    Analyze market efficiency and performance metrics.
    """
    eval_df = df_rounds[df_rounds['round'] >= eval_start_round].copy()
    
    metrics = {
        'Market Efficiency': f"{eval_df['market_efficiency'].mean():.3f} ({eval_df['market_efficiency'].std():.3f})",
        'Avg Trades': f"{eval_df['actual_trades'].mean():.1f} ({eval_df['actual_trades'].std():.1f})",
        'Avg Total Profit': f"{eval_df['actual_total_profit'].mean():.1f} ({eval_df['actual_total_profit'].std():.1f})",
        'Price Deviation': f"{eval_df['abs_diff_price'].mean():.1f} ({eval_df['abs_diff_price'].std():.1f})",
        'Quantity Deviation': f"{eval_df['abs_diff_quantity'].mean():.1f} ({eval_df['abs_diff_quantity'].std():.1f})",
    }
    
    return pd.DataFrame([metrics])


def perform_statistical_tests(df_rounds, eval_start_round):
    """
    Perform statistical significance tests between strategies.
    """
    eval_df = df_rounds[df_rounds['round'] >= eval_start_round].copy()
    
    # Collect profits by strategy
    strategy_profits = {}
    for _, row in eval_df.iterrows():
        bot_details = ast.literal_eval(row['bot_details'])
        for bot in bot_details:
            strategy = bot['strategy']
            if strategy not in strategy_profits:
                strategy_profits[strategy] = []
            strategy_profits[strategy].append(bot['profit'])
    
    # Perform pairwise t-tests
    strategies = list(strategy_profits.keys())
    test_results = []
    
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            strat1, strat2 = strategies[i], strategies[j]
            profits1 = strategy_profits[strat1]
            profits2 = strategy_profits[strat2]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(profits1, profits2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(profits1)**2 + np.std(profits2)**2) / 2)
            cohen_d = (np.mean(profits1) - np.mean(profits2)) / pooled_std if pooled_std > 0 else 0
            
            test_results.append({
                'Comparison': f"{strat1} vs {strat2}",
                'Mean Diff': f"{np.mean(profits1) - np.mean(profits2):.2f}",
                'p-value': f"{p_value:.4f}",
                'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns',
                'Cohen\'s d': f"{cohen_d:.3f}",
            })
    
    return pd.DataFrame(test_results)


def analyze_learning_progression(df_rounds, training_end_round):
    """
    Analyze how agent performance improves during training.
    """
    train_df = df_rounds[df_rounds['round'] < training_end_round].copy()
    
    # Divide training into phases
    n_phases = 5
    phase_size = len(train_df) // n_phases
    
    phase_stats = []
    for phase in range(n_phases):
        start_idx = phase * phase_size
        end_idx = (phase + 1) * phase_size if phase < n_phases - 1 else len(train_df)
        phase_df = train_df.iloc[start_idx:end_idx]
        
        # Collect PPO profits
        ppo_profits = []
        other_profits = []
        
        for _, row in phase_df.iterrows():
            bot_details = ast.literal_eval(row['bot_details'])
            for bot in bot_details:
                if 'ppo' in bot['strategy'].lower():
                    ppo_profits.append(bot['profit'])
                else:
                    other_profits.append(bot['profit'])
        
        phase_stats.append({
            'Phase': f"{phase+1}/{n_phases}",
            'Rounds': f"{phase_df['round'].min()}-{phase_df['round'].max()}",
            'PPO Mean': f"{np.mean(ppo_profits) if ppo_profits else 0:.2f}",
            'Other Mean': f"{np.mean(other_profits) if other_profits else 0:.2f}",
            'PPO Win Rate': f"{sum(1 for p in ppo_profits if p > 0) / len(ppo_profits) * 100 if ppo_profits else 0:.1f}%"
        })
    
    return pd.DataFrame(phase_stats)


def main():
    """
    Main analysis function - runs all tournament analyses.
    """
    print("\n" + "="*80)
    print("PPO HANDCRAFTED TRADER - TOURNAMENT ANALYSIS")
    print("="*80 + "\n")
    
    # Find all test results
    test_dirs = {
        'ZIC': 'experiments/ppo_handcrafted_tests/test_ppo_handcrafted_vs_zic',
        'ZIP': 'experiments/ppo_handcrafted_tests/test_ppo_handcrafted_vs_zip',
        'GD': 'experiments/ppo_handcrafted_tests/test_ppo_vs_gd_long',
        'Kaplan': 'experiments/ppo_handcrafted_tests/test_ppo_vs_kaplan_long',
        'Mixed': 'experiments/ppo_handcrafted_tests/test_ppo_vs_mixed_long',
    }
    
    all_results = []
    
    for opponent, test_dir in test_dirs.items():
        round_log = Path(test_dir) / "round_log_all.csv"
        
        if not round_log.exists():
            print(f"⚠️  No results found for {opponent} test")
            continue
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: PPOHandcrafted vs {opponent}")
        print(f"{'='*60}")
        
        # Load data
        df_rounds = pd.read_csv(round_log)
        
        # Determine training/eval split
        if 'kaplan' in test_dir.lower():
            eval_start = 1800
        elif 'long' in test_dir:
            eval_start = 1300
        elif 'zic' in test_dir or 'zip' in test_dir:
            eval_start = 400 if 'test_ppo_handcrafted' in test_dir else 900
        else:
            eval_start = int(len(df_rounds) * 0.8)
        
        # 1. Strategy Tournament Ranking
        print("\n1. STRATEGY TOURNAMENT RANKING")
        print("-" * 40)
        tournament_df = analyze_strategy_tournament(df_rounds, eval_start)
        print(tournament_df.to_string(index=False))
        
        # Check if PPO wins
        if len(tournament_df) > 0 and 'ppo' in tournament_df.iloc[0]['Strategy'].lower():
            print("\n✅ PPOHandcrafted WINS this matchup!")
        
        # 2. Market Efficiency
        print("\n2. MARKET EFFICIENCY METRICS")
        print("-" * 40)
        efficiency_df = analyze_market_efficiency(df_rounds, eval_start)
        print(efficiency_df.to_string(index=False))
        
        # 3. Statistical Significance
        print("\n3. STATISTICAL SIGNIFICANCE TESTS")
        print("-" * 40)
        stats_df = perform_statistical_tests(df_rounds, eval_start)
        print(stats_df.to_string(index=False))
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # 4. Learning Progression
        if eval_start > 0:
            print("\n4. LEARNING PROGRESSION")
            print("-" * 40)
            learning_df = analyze_learning_progression(df_rounds, eval_start)
            print(learning_df.to_string(index=False))
        
        # Store results
        all_results.append({
            'opponent': opponent,
            'tournament': tournament_df,
            'efficiency': efficiency_df,
            'stats': stats_df
        })
    
    # Final Summary
    if all_results:
        print("\n" + "="*80)
        print("FINAL TOURNAMENT SUMMARY")
        print("="*80)
        
        wins = 0
        for result in all_results:
            if len(result['tournament']) > 0:
                winner = result['tournament'].iloc[0]['Strategy']
                if 'ppo' in winner.lower():
                    wins += 1
                    print(f"✅ vs {result['opponent']}: PPOHandcrafted WINS")
                else:
                    print(f"❌ vs {result['opponent']}: {winner} wins")
        
        print(f"\n🏆 PPOHandcrafted won {wins}/{len(all_results)} matchups")
        
        if wins >= len(all_results) * 0.5:
            print("\n🎉 SUCCESS: PPOHandcraftedTrader demonstrates clear learning ability!")
            print("The agent successfully beats classical trading strategies.")
        else:
            print("\n⚠️  PPOHandcrafted needs more tuning or training time")
        
        # Save results to JSON for paper tables
        output_file = "tournament_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': f"PPOHandcrafted won {wins}/{len(all_results)} matchups",
                'details': [
                    {
                        'opponent': r['opponent'],
                        'winner': r['tournament'].iloc[0]['Strategy'] if len(r['tournament']) > 0 else 'N/A',
                        'efficiency': r['efficiency'].iloc[0]['Market Efficiency'] if len(r['efficiency']) > 0 else 'N/A'
                    } 
                    for r in all_results
                ]
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
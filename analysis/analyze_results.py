#!/usr/bin/env python3
"""Analyze PPOHandcrafted test results"""

import pandas as pd
import numpy as np
import ast

# Load the round data
df = pd.read_csv('experiments/ppo_handcrafted_tests/test_ppo_handcrafted_vs_zic/round_log_all.csv')

# Filter for evaluation rounds (80-99)
eval_df = df[df['round'] >= 80].copy()

# Extract profit data from bot_details column
profits_by_agent = {'PPO_B0': [], 'PPO_B1': [], 'ZIC_B2': [], 'ZIC_B3': [], 'ZIC_B4': [], 'ZIC_B5': []}
trades_by_agent = {'PPO_B0': [], 'PPO_B1': [], 'ZIC_B2': [], 'ZIC_B3': [], 'ZIC_B4': [], 'ZIC_B5': []}

for _, row in eval_df.iterrows():
    bot_details = ast.literal_eval(row['bot_details'])
    for bot in bot_details[:6]:  # Only buyers
        if bot['name'] == 'B0':
            profits_by_agent['PPO_B0'].append(bot['profit'])
            trades_by_agent['PPO_B0'].append(bot['trades'])
        elif bot['name'] == 'B1':
            profits_by_agent['PPO_B1'].append(bot['profit'])
            trades_by_agent['PPO_B1'].append(bot['trades'])
        elif bot['name'] == 'B2':
            profits_by_agent['ZIC_B2'].append(bot['profit'])
            trades_by_agent['ZIC_B2'].append(bot['trades'])
        elif bot['name'] == 'B3':
            profits_by_agent['ZIC_B3'].append(bot['profit'])
            trades_by_agent['ZIC_B3'].append(bot['trades'])
        elif bot['name'] == 'B4':
            profits_by_agent['ZIC_B4'].append(bot['profit'])
            trades_by_agent['ZIC_B4'].append(bot['trades'])
        elif bot['name'] == 'B5':
            profits_by_agent['ZIC_B5'].append(bot['profit'])
            trades_by_agent['ZIC_B5'].append(bot['trades'])

# Calculate statistics
print('\n' + '='*70)
print('PPOHandcrafted vs ZIC - Final 20 Evaluation Rounds (80-99)')
print('='*70)
print('\nBUYER AGENTS PERFORMANCE:')
print('-' * 70)
print(f'{"Agent":<8} | {"Type":<15} | {"Mean Profit":<12} | {"Max":>5} | {"Trades":>6} | {"Win%":>5}')
print('-' * 70)

ppo_profits = []
zic_profits = []
ppo_trades = 0
zic_trades = 0

for agent in ['PPO_B0', 'PPO_B1', 'ZIC_B2', 'ZIC_B3', 'ZIC_B4', 'ZIC_B5']:
    profits = profits_by_agent[agent]
    trades = trades_by_agent[agent]
    agent_type = 'PPOHandcrafted' if 'PPO' in agent else 'ZIC'
    mean_profit = np.mean(profits)
    max_profit = np.max(profits) if profits else 0
    total_trades = sum(trades)
    win_rate = (sum(1 for p in profits if p > 0) / len(profits) * 100) if profits else 0
    
    if 'PPO' in agent:
        ppo_profits.extend(profits)
        ppo_trades += total_trades
    else:
        zic_profits.extend(profits)
        zic_trades += total_trades
    
    print(f'{agent:<8} | {agent_type:<15} | ${mean_profit:>10.2f}  | ${max_profit:>4.0f} | {total_trades:>6} | {win_rate:>4.0f}%')

print('-' * 70)

# Summary statistics
ppo_mean = np.mean(ppo_profits) if ppo_profits else 0
zic_mean = np.mean(zic_profits) if zic_profits else 0
ppo_total = sum(ppo_profits)
zic_total = sum(zic_profits)

print(f'\nAGGREGATE RESULTS:')
print(f'{"Strategy":<15} | {"Total Profit":<12} | {"Mean/Agent":<12} | {"Total Trades":<12}')
print('-' * 55)
print(f'{"PPOHandcrafted":<15} | ${ppo_total:>10.2f}  | ${ppo_mean:>10.2f}  | {ppo_trades:>12}')
print(f'{"ZIC":<15} | ${zic_total:>10.2f}  | ${zic_mean:>10.2f}  | {zic_trades:>12}')

print('\n' + '='*70)
if ppo_mean > zic_mean:
    improvement = ((ppo_mean - zic_mean) / max(abs(zic_mean), 0.01)) * 100 if zic_mean != 0 else 100
    print(f'✓ WINNER: PPOHandcrafted ({improvement:.0f}% better per agent)')
elif zic_mean > ppo_mean:
    deficit = ((zic_mean - ppo_mean) / max(abs(ppo_mean), 0.01)) * 100 if ppo_mean != 0 else 100
    print(f'✗ WINNER: ZIC ({deficit:.0f}% better per agent)')
else:
    print('TIE: Both strategies performed equally')
print('='*70)
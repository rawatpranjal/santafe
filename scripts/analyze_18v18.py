import pandas as pd
import sys

# Load 18v18 results
df = pd.read_csv("results/grand_melee_18v18/results.csv")

# Group by agent type
ranking = df.groupby('agent_type').agg({
    'period_profit': ['mean', 'std'],
    'num_trades': 'mean',
    'efficiency': 'mean'
}).round(2)

ranking.columns = ['profit_mean', 'profit_std', 'trades_mean', 'efficiency']
ranking = ranking.sort_values('profit_mean', ascending=False)
ranking['rank'] = range(1, len(ranking) + 1)

print("\n" + "="*80)
print("GRAND MELEE 18v18 RESULTS")
print("="*80)
print(ranking[['profit_mean', 'profit_std', 'trades_mean', 'efficiency', 'rank']])

# Compare to 9v9
try:
    df_9v9 = pd.read_csv("results/grand_melee_9v9_v2/results.csv")
    ranking_9v9 = df_9v9.groupby('agent_type').agg({'period_profit': 'mean'}).round(2)
    ranking_9v9 = ranking_9v9.sort_values('period_profit', ascending=False)
    ranking_9v9['rank_9v9'] = range(1, len(ranking_9v9) + 1)
    
    print("\n" + "="*80)
    print("RANK COMPARISON: 9v9 vs 18v18")
    print("="*80)
    
    comparison = pd.DataFrame({
        'profit_9v9': ranking_9v9['period_profit'],
        'rank_9v9': ranking_9v9['rank_9v9'],
        'profit_18v18': ranking['profit_mean'],
        'rank_18v18': ranking['rank']
    })
    print(comparison)
except Exception as e:
    print(f"Could not load 9v9 for comparison: {e}")

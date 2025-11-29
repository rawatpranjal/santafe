import pandas as pd
import sys

def analyze_grand_melee(results_path):
    print(f"Loading results from {results_path}")
    df = pd.read_csv(results_path)
    
    # Group by agent type and calculate metrics
    ranking = df.groupby('agent_type').agg({
        'period_profit': ['mean', 'std', 'min', 'max'],
        'num_trades': 'mean',
        'efficiency': 'mean'
    }).round(2)
    
    # Flatten columns
    ranking.columns = ['_'.join(col).strip() for col in ranking.columns.values]
    
    # Sort by mean profit
    ranking = ranking.sort_values('period_profit_mean', ascending=False)
    
    # Add rank
    ranking['rank'] = range(1, len(ranking) + 1)
    
    print("\n" + "="*80)
    print("GRAND MELEE RANKINGS (Sorted by Individual Profit)")
    print("="*80)
    print(ranking[['period_profit_mean', 'period_profit_std', 'num_trades_mean', 'efficiency_mean', 'rank']])
    
    # Check for ZI (should be excluded if not in config, but good to verify)
    if 'ZI' in ranking.index:
        print("\nNote: ZI is present in the rankings.")
        
    return ranking

if __name__ == "__main__":
    results_path = "results/grand_melee_9v9_v2/results.csv"
    analyze_grand_melee(results_path)

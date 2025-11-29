import pandas as pd
import sys

def analyze_scenario(results_path, scenario_name):
    """Analyze a single market scenario."""
    try:
        df = pd.read_csv(results_path)
        ranking = df.groupby('agent_type').agg({
            'period_profit': ['mean', 'std'],
            'efficiency': 'mean'
        }).round(2)
        
        ranking.columns = ['profit_mean', 'profit_std', 'efficiency']
        ranking = ranking.sort_values('profit_mean', ascending=False)
        ranking['rank'] = range(1, len(ranking) + 1)
        
        return ranking
    except Exception as e:
        print(f"Error loading {scenario_name}: {e}")
        return None

if __name__ == "__main__":
    print("="*80)
    print("MARKET VARIATION ANALYSIS")
    print("="*80)
    
    scenarios = [
        ("results/grand_melee_asymmetric_6v3/results.csv", "ASYMMETRIC (6v3)"),
        ("results/grand_melee_scarcity/results.csv", "TOKEN SCARCITY (2 tokens)"),
        ("results/grand_melee_time_pressure/results.csv", "TIME PRESSURE (50 steps)"),
        ("results/grand_melee_extreme_8v1/results.csv", "EXTREME ASYMMETRY (8v1)"),
        ("results/grand_melee_ultra_pressure/results.csv", "ULTRA PRESSURE (25 steps)"),
        ("results/grand_melee_minimal_tokens/results.csv", "MINIMAL TOKENS (1 token)"),
        ("results/grand_melee_long_periods/results.csv", "LONG PERIODS (20 periods)"),
        ("results/grand_melee_short_periods/results.csv", "SHORT PERIODS (3 periods)"),
        ("results/grand_melee_long_steps/results.csv", "LONG STEPS (200 steps)"),
        ("results/grand_melee_ultra_short_steps/results.csv", "ULTRA SHORT STEPS (10 steps)"),
    ]
    
    all_rankings = {}
    
    for path, name in scenarios:
        print(f"\n{name}")
        print("-"*80)
        ranking = analyze_scenario(path, name)
        if ranking is not None:
            print(ranking[['profit_mean', 'efficiency', 'rank']])
            all_rankings[name] = ranking
    
    # Cross-scenario comparison
    print("\n" + "="*80)
    print("RANK STABILITY ACROSS SCENARIOS")
    print("="*80)
    
    for agent in ['Jacobson', 'Perry', 'Kaplan', 'Skeleton', 'Lin', 'GD', 'ZI2', 'ZIC', 'ZIP']:
        ranks = []
        for scenario_name, ranking in all_rankings.items():
            if agent in ranking.index:
                ranks.append(int(ranking.loc[agent, 'rank']))
        if ranks:
            print(f"{agent:12} Ranks: {ranks}   Range: {max(ranks)-min(ranks)}   Avg: {sum(ranks)/len(ranks):.1f}")

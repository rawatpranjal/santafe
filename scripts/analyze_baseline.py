"""
Analyze 1993 Baseline Tournament Results.

Compares profitability across agent types (ZIC, Kaplan, ZIP, GD) to:
1. Validate overall market efficiency
2. Measure individual agent profitability
3. Determine if Kaplan's 11-14% advantage over ZIC is historically accurate
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_tournament(results_csv: Path):
    """Analyze tournament results and print per-agent profitability."""

    df = pd.read_csv(results_csv)

    print("=" * 80)
    print("1993 BASELINE TOURNAMENT ANALYSIS")
    print("=" * 80)
    print(f"\nTotal periods: {len(df)}")
    print(f"Unique agents: {df['agent_id'].nunique()}")
    print(f"Agent types: {df['agent_type'].unique().tolist()}")

    # Overall efficiency
    print("\n" + "=" * 80)
    print("OVERALL MARKET EFFICIENCY")
    print("=" * 80)
    eff_stats = df.groupby(['round', 'period'])['efficiency'].first().describe()
    print(eff_stats)

    # Per-agent profitability
    print("\n" + "=" * 80)
    print("PER-AGENT PROFITABILITY")
    print("=" * 80)

    agent_stats = df.groupby(['agent_type', 'is_buyer']).agg({
        'period_profit': ['count', 'mean', 'std', 'min', 'max', 'sum'],
        'num_trades': ['mean', 'sum']
    }).round(2)

    print(agent_stats)

    # Agent type comparison (buyers)
    print("\n" + "=" * 80)
    print("BUYER PROFITABILITY COMPARISON")
    print("=" * 80)

    buyers = df[df['is_buyer'] == True].groupby('agent_type').agg({
        'period_profit': ['mean', 'std', 'count'],
        'num_trades': 'mean'
    }).round(2)

    buyers.columns = ['Mean_Profit', 'Std_Profit', 'Periods', 'Mean_Trades']
    buyers = buyers.sort_values('Mean_Profit', ascending=False)
    print(buyers)

    # Calculate profit ratios vs ZIC
    if 'ZIC' in buyers.index:
        zic_profit = buyers.loc['ZIC', 'Mean_Profit']
        buyers['Ratio_vs_ZIC'] = (buyers['Mean_Profit'] / zic_profit).round(3)
        buyers['Advantage_%'] = ((buyers['Mean_Profit'] / zic_profit - 1) * 100).round(1)
        print("\n**Buyer Profit Ratios vs ZIC:**")
        print(buyers[['Mean_Profit', 'Ratio_vs_ZIC', 'Advantage_%']])

    # Agent type comparison (sellers)
    print("\n" + "=" * 80)
    print("SELLER PROFITABILITY COMPARISON")
    print("=" * 80)

    sellers = df[df['is_buyer'] == False].groupby('agent_type').agg({
        'period_profit': ['mean', 'std', 'count'],
        'num_trades': 'mean'
    }).round(2)

    sellers.columns = ['Mean_Profit', 'Std_Profit', 'Periods', 'Mean_Trades']
    sellers = sellers.sort_values('Mean_Profit', ascending=False)
    print(sellers)

    # Calculate profit ratios vs ZIC
    if 'ZIC' in sellers.index:
        zic_profit = sellers.loc['ZIC', 'Mean_Profit']
        sellers['Ratio_vs_ZIC'] = (sellers['Mean_Profit'] / zic_profit).round(3)
        sellers['Advantage_%'] = ((sellers['Mean_Profit'] / zic_profit - 1) * 100).round(1)
        print("\n**Seller Profit Ratios vs ZIC:**")
        print(sellers[['Mean_Profit', 'Ratio_vs_ZIC', 'Advantage_%']])

    # Kaplan-specific analysis
    print("\n" + "=" * 80)
    print("KAPLAN VALIDATION")
    print("=" * 80)

    if 'Kaplan' in buyers.index and 'ZIC' in buyers.index:
        kaplan_buyer_profit = buyers.loc['Kaplan', 'Mean_Profit']
        zic_buyer_profit = buyers.loc['ZIC', 'Mean_Profit']
        buyer_ratio = kaplan_buyer_profit / zic_buyer_profit
        buyer_adv = (buyer_ratio - 1) * 100

        print(f"Buyer: Kaplan {kaplan_buyer_profit:.2f} vs ZIC {zic_buyer_profit:.2f}")
        print(f"  Ratio: {buyer_ratio:.3f} ({buyer_adv:+.1f}%)")

    if 'Kaplan' in sellers.index and 'ZIC' in sellers.index:
        kaplan_seller_profit = sellers.loc['Kaplan', 'Mean_Profit']
        zic_seller_profit = sellers.loc['ZIC', 'Mean_Profit']
        seller_ratio = kaplan_seller_profit / zic_seller_profit
        seller_adv = (seller_ratio - 1) * 100

        print(f"Seller: Kaplan {kaplan_seller_profit:.2f} vs ZIC {zic_seller_profit:.2f}")
        print(f"  Ratio: {seller_ratio:.3f} ({seller_adv:+.1f}%)")

        avg_ratio = (buyer_ratio + seller_ratio) / 2
        avg_adv = (avg_ratio - 1) * 100
        print(f"\nAverage: {avg_ratio:.3f} ({avg_adv:+.1f}%)")

        if avg_adv < 10:
            print("  ⚠️  WARNING: Kaplan advantage < 10% (below expected 20-40%)")
        elif avg_adv < 20:
            print("  ⚠️  MARGINAL: Kaplan advantage 10-20% (below expected 20-40%)")
        else:
            print("  ✅ ACCEPTABLE: Kaplan advantage >= 20%")

    # Population share vs profit share
    print("\n" + "=" * 80)
    print("PROFIT SHARE ANALYSIS")
    print("=" * 80)

    for is_buyer in [True, False]:
        role = "BUYER" if is_buyer else "SELLER"
        role_df = df[df['is_buyer'] == is_buyer]

        # Count agents of each type
        agent_counts = role_df.groupby('agent_type')['agent_id'].nunique()
        total_agents = agent_counts.sum()

        # Total profit by type
        profit_by_type = role_df.groupby('agent_type')['period_profit'].sum()
        total_profit = profit_by_type.sum()

        # Calculate shares
        pop_share = (agent_counts / total_agents * 100).round(1)
        profit_share = (profit_by_type / total_profit * 100).round(1)

        comparison = pd.DataFrame({
            'Population_Share_%': pop_share,
            'Profit_Share_%': profit_share,
            'Diff': profit_share - pop_share
        }).sort_values('Profit_Share_%', ascending=False)

        print(f"\n{role}S:")
        print(comparison)


if __name__ == "__main__":
    results_path = Path("results/tournament_1993_baseline/results.csv")

    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        print("Please run the tournament first:")
        print("  python scripts/run_experiment.py experiment=tournament_1993_baseline")
    else:
        analyze_tournament(results_path)

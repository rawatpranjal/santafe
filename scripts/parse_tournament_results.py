#!/usr/bin/env python3
"""
Parse tournament results and generate markdown tables for tracker.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

def parse_pure_markets():
    """Parse pure market self-play efficiency results."""
    pure_dir = Path("results/tournament_pure_20251124_032925/pure")

    results = []
    for trader_dir in sorted(pure_dir.iterdir()):
        if not trader_dir.is_dir():
            continue

        csv_file = trader_dir / "results.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Get trader name from directory
        trader_name = trader_dir.name.replace("pure_", "").upper()

        # Calculate efficiency stats (aggregate across periods)
        efficiency_mean = df['efficiency'].mean()
        efficiency_std = df['efficiency'].std()
        num_periods = df['period'].nunique()

        results.append({
            'trader': trader_name,
            'efficiency_mean': efficiency_mean,
            'efficiency_std': efficiency_std,
            'num_periods': num_periods
        })

    # Sort by efficiency descending
    results = sorted(results, key=lambda x: x['efficiency_mean'], reverse=True)
    return results

def parse_pairwise_tournaments():
    """Parse pairwise tournament results."""
    pairwise_dir = Path("results/tournament_pairwise_20251124_032931/pairwise")

    results = []
    for matchup_dir in sorted(pairwise_dir.iterdir()):
        if not matchup_dir.is_dir():
            continue

        csv_file = matchup_dir / "results.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Get matchup name
        matchup_name = matchup_dir.name.replace("_", " ").title()

        # Calculate efficiency and profit shares
        efficiency_mean = df['efficiency'].mean()
        efficiency_std = df['efficiency'].std()
        num_periods = df['period'].nunique()

        # Calculate profit shares by agent type
        profit_by_type = df.groupby('agent_type')['period_profit'].sum()
        total_profit = profit_by_type.sum()

        if len(profit_by_type) >= 2:
            trader1, trader2 = sorted(profit_by_type.index)
            trader1_share = (profit_by_type[trader1] / total_profit * 100) if total_profit > 0 else 50
            trader2_share = (profit_by_type[trader2] / total_profit * 100) if total_profit > 0 else 50
            winner = trader1 if trader1_share > trader2_share else trader2
            winner_share = max(trader1_share, trader2_share)
        else:
            winner = "N/A"
            winner_share = 0

        results.append({
            'matchup': matchup_name,
            'efficiency_mean': efficiency_mean,
            'efficiency_std': efficiency_std,
            'winner': winner,
            'winner_share': winner_share,
            'num_periods': num_periods
        })

    return results

def parse_1v7_invasibility():
    """Parse 1v7 invasibility test results."""
    base_dir = Path("results/tournament_1v7_20251124_033015/one_v_seven")

    if not base_dir.exists():
        return []

    trader_data = {}

    # Iterate through all 1v7 result directories
    for config_dir in sorted(base_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        csv_file = config_dir / "results.csv"
        if not csv_file.exists():
            continue

        # Parse config name (e.g., "gd_1v7_zic", "jacobson_1v7_mixed")
        config_name = config_dir.name
        parts = config_name.split("_")
        if len(parts) < 2:
            continue

        trader = parts[0].upper()

        # Read efficiency data
        df = pd.read_csv(csv_file)
        efficiency_mean = df['efficiency'].mean()
        efficiency_std = df['efficiency'].std()
        num_periods = df['period'].nunique()

        # Determine if this is buyer or seller test based on agent data
        trader_rows = df[df['agent_type'].str.lower() == trader.lower()]
        if trader_rows.empty:
            continue

        is_buyer = trader_rows['is_buyer'].iloc[0]

        # Initialize trader entry if needed
        if trader not in trader_data:
            trader_data[trader] = {
                'trader': trader,
                'buyer_eff_mean': 0,
                'buyer_eff_std': 0,
                'seller_eff_mean': 0,
                'seller_eff_std': 0,
                'num_periods': num_periods
            }

        # Store results based on role
        if is_buyer:
            trader_data[trader]['buyer_eff_mean'] = efficiency_mean
            trader_data[trader]['buyer_eff_std'] = efficiency_std
        else:
            trader_data[trader]['seller_eff_mean'] = efficiency_mean
            trader_data[trader]['seller_eff_std'] = efficiency_std

    # Convert to list and calculate invasibility
    results = list(trader_data.values())
    for entry in results:
        entry['invasibility'] = (entry['buyer_eff_mean'] + entry['seller_eff_mean']) / 2

    # Sort by invasibility descending
    results = sorted(results, key=lambda x: x['invasibility'], reverse=True)
    return results

def parse_mixed_markets():
    """Parse mixed market results with varying Kaplan percentages."""
    mixed_dir = Path("results/tournament_mixed_20251124_033011/mixed")

    if not mixed_dir.exists():
        return []

    results = []

    # Iterate through all mixed result directories
    for config_dir in sorted(mixed_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        csv_file = config_dir / "results.csv"
        if not csv_file.exists():
            continue

        # Parse config name to get Kaplan percentage
        # Format: "kaplan_background_XXpct"
        config_name = config_dir.name
        if 'kaplan_background' not in config_name:
            continue

        parts = config_name.split("_")
        kaplan_pct = None
        for part in parts:
            if part.endswith('pct'):
                try:
                    kaplan_pct = int(part.replace('pct', ''))
                    break
                except ValueError:
                    continue

        if kaplan_pct is None:
            continue

        # Read efficiency data
        df = pd.read_csv(csv_file)
        efficiency_mean = df['efficiency'].mean()
        efficiency_std = df['efficiency'].std()
        num_periods = df['period'].nunique()

        results.append({
            'kaplan_pct': kaplan_pct,
            'efficiency_mean': efficiency_mean,
            'efficiency_std': efficiency_std,
            'num_periods': num_periods
        })

    # Sort by Kaplan percentage
    results = sorted(results, key=lambda x: x['kaplan_pct'])
    return results

def generate_table5_markdown(pure_results):
    """Generate Table 5: Pure Market Self-Play Efficiency"""
    lines = []
    lines.append("### Table 5: Pure Market Self-Play Efficiency")
    lines.append("")
    lines.append("| Trader   | Efficiency (mean ± std) | Periods | Interpretation                  |")
    lines.append("|----------|-------------------------|---------|---------------------------------|")

    for r in pure_results:
        eff_str = f"{r['efficiency_mean']:.1f}% ± {r['efficiency_std']:.1f}%"

        # Add interpretation based on efficiency
        if r['efficiency_mean'] >= 99:
            interp = "Near-optimal self-play"
        elif r['efficiency_mean'] >= 95:
            interp = "Excellent self-play"
        elif r['efficiency_mean'] >= 90:
            interp = "Good self-play"
        elif r['efficiency_mean'] >= 80:
            interp = "Moderate self-play"
        else:
            interp = "Poor self-play (market failure?)"

        lines.append(f"| {r['trader']:<8} | {eff_str:<23} | {r['num_periods']:<7} | {interp:<31} |")

    lines.append("")
    return "\n".join(lines)

def generate_table7_markdown(pairwise_results):
    """Generate Table 7: Pairwise Tournament Results"""
    lines = []
    lines.append("### Table 7: Pairwise Tournament Results")
    lines.append("")
    lines.append("| Matchup              | Efficiency (mean ± std) | Winner Profit Share | Periods | Notes           |")
    lines.append("|----------------------|-------------------------|---------------------|---------|-----------------|")

    for r in pairwise_results:
        eff_str = f"{r['efficiency_mean']:.1f}% ± {r['efficiency_std']:.1f}%"
        share_str = f"{r['winner_share']:.1f}%"

        # Add note based on winner dominance
        if r['winner_share'] > 70:
            note = "Strong dominance"
        elif r['winner_share'] > 60:
            note = "Moderate dominance"
        elif r['winner_share'] > 55:
            note = "Slight advantage"
        else:
            note = "Balanced"

        lines.append(f"| {r['matchup']:<20} | {eff_str:<23} | {share_str:<19} | {r['num_periods']:<7} | {note:<15} |")

    lines.append("")
    return "\n".join(lines)

def generate_table8_markdown(invasibility_results):
    """Generate Table 8: Complete 1v7 Invasibility Matrix"""
    lines = []
    lines.append("### Table 8: Complete 1v7 Invasibility Matrix")
    lines.append("")
    lines.append("*Test: 1 trader (varied) vs 7 ZIC agents. Measures individual trader's ability to invade/exploit ZIC population.*")
    lines.append("")
    lines.append("| Trader   | Invasibility | As Buyer         | As Seller        | Interpretation                |")
    lines.append("|----------|--------------|------------------|------------------|-------------------------------|")

    for r in invasibility_results:
        inv_str = f"{r['invasibility']:.1f}%"
        buyer_str = f"{r['buyer_eff_mean']:.1f}% ± {r['buyer_eff_std']:.1f}%"
        seller_str = f"{r['seller_eff_mean']:.1f}% ± {r['seller_eff_std']:.1f}%"

        # Add interpretation
        if r['invasibility'] >= 99:
            interp = "Excellent invasibility"
        elif r['invasibility'] >= 95:
            interp = "Strong invasibility"
        elif r['invasibility'] >= 90:
            interp = "Good invasibility"
        elif r['invasibility'] >= 80:
            interp = "Moderate invasibility"
        else:
            interp = "Weak invasibility"

        lines.append(f"| {r['trader']:<8} | {inv_str:<12} | {buyer_str:<16} | {seller_str:<16} | {interp:<29} |")

    lines.append("")
    return "\n".join(lines)

def generate_table9_markdown(mixed_results):
    """Generate Table 9: Mixed Market Kaplan Background Effect"""
    lines = []
    lines.append("### Table 9: Mixed Market Efficiency (Kaplan Background Effect)")
    lines.append("")
    lines.append("*Hypothesis: Kaplan-dominated markets crash (expect <60% efficiency at high %)*")
    lines.append("")
    lines.append("| Kaplan % | Efficiency (mean ± std) | Periods | Interpretation                     |")
    lines.append("|----------|-------------------------|---------|------------------------------------|")

    for r in mixed_results:
        eff_str = f"{r['efficiency_mean']:.1f}% ± {r['efficiency_std']:.1f}%"

        # Add interpretation based on Kaplan percentage and efficiency
        if r['kaplan_pct'] == 0:
            interp = "Baseline (no Kaplan)"
        elif r['kaplan_pct'] <= 25:
            interp = "Low Kaplan concentration"
        elif r['kaplan_pct'] <= 50:
            if r['efficiency_mean'] < 80:
                interp = "Moderate - efficiency declining"
            else:
                interp = "Moderate concentration"
        elif r['kaplan_pct'] <= 75:
            if r['efficiency_mean'] < 60:
                interp = "High - market failure observed"
            else:
                interp = "High concentration"
        else:
            if r['efficiency_mean'] < 60:
                interp = "Near-homogeneous - CRASH confirmed"
            else:
                interp = "Near-homogeneous Kaplan market"

        lines.append(f"| {r['kaplan_pct']:<8} | {eff_str:<23} | {r['num_periods']:<7} | {interp:<34} |")

    lines.append("")
    return "\n".join(lines)

def main():
    print("Parsing tournament results...")

    # Parse all tournament data
    pure_results = parse_pure_markets()
    print(f"  Pure markets: {len(pure_results)} traders")

    pairwise_results = parse_pairwise_tournaments()
    print(f"  Pairwise: {len(pairwise_results)} matchups")

    invasibility_results = parse_1v7_invasibility()
    print(f"  1v7 invasibility: {len(invasibility_results)} traders")

    mixed_results = parse_mixed_markets()
    print(f"  Mixed markets: {len(mixed_results)} configs")

    # Generate markdown tables
    print("\nGenerating markdown tables...")

    table5 = generate_table5_markdown(pure_results)
    table7 = generate_table7_markdown(pairwise_results)
    table8 = generate_table8_markdown(invasibility_results)
    table9 = generate_table9_markdown(mixed_results)

    # Write to output file
    output_file = Path("tournament_tables_output.md")
    with open(output_file, "w") as f:
        f.write("# Tournament Results Tables\n\n")
        f.write(table5)
        f.write("\n")
        f.write(table7)
        f.write("\n")
        f.write(table8)
        f.write("\n")
        f.write(table9)

    print(f"\nTables written to: {output_file}")
    print("\nPreview:")
    print("=" * 80)
    print(table5)

if __name__ == "__main__":
    main()

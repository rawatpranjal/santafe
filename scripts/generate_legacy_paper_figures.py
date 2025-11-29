#!/usr/bin/env python3
"""
Generate tables and figures for legacy trader performance paper section.

Processes tournament results from:
- Pure self-play
- Invasibility (1v7 vs ZIC)
- Grand Melee (10 scenarios)

Outputs:
- LaTeX tables
- PDF figures
- Summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from glob import glob

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (6, 4)

# Output directories
PAPER_DIR = Path("paper/arxiv")
FIGURES_DIR = PAPER_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Trader display names and colors
TRADER_NAMES = {
    'ZIC': 'ZIC',
    'ZI': 'ZI',
    'ZI2': 'ZI2',
    'ZIP': 'ZIP',
    'GD': 'GD',
    'Kaplan': 'Kaplan',
    'Lin': 'Lin',
    'Jacobson': 'Jacobson',
    'Skeleton': 'Skeleton',
    'Perry': 'Perry'
}

TRADER_COLORS = {
    'ZIC': '#95a5a6',     # Gray (baseline)
    'ZI': '#bdc3c7',      # Light gray (control)
    'ZI2': '#3498db',     # Blue (good)
    'ZIP': '#e74c3c',     # Red (problematic)
    'GD': '#2ecc71',      # Green (excellent)
    'Kaplan': '#f39c12',  # Orange (strategic)
    'Lin': '#9b59b6',     # Purple (efficient)
    'Jacobson': '#1abc9c', # Teal (robust)
    'Skeleton': '#34495e', # Dark gray
    'Perry': '#e67e22'    # Orange-red (strong)
}


def load_grand_melee_data():
    """Load all Grand Melee scenario results."""
    scenarios = [
        ("results/grand_melee_baseline/results.csv", "Baseline (9v9)"),
        ("results/grand_melee_asymmetric_6v3/results.csv", "Asymmetric (6v3)"),
        ("results/grand_melee_scarcity/results.csv", "Token Scarcity (2)"),
        ("results/grand_melee_minimal_tokens/results.csv", "Minimal Tokens (1)"),
        ("results/grand_melee_time_pressure/results.csv", "Time Pressure (50s)"),
        ("results/grand_melee_ultra_pressure/results.csv", "Ultra Pressure (25s)"),
        ("results/grand_melee_ultra_short_steps/results.csv", "Ultra Short (10s)"),
        ("results/grand_melee_long_steps/results.csv", "Long Steps (200s)"),
        ("results/grand_melee_long_periods/results.csv", "Long Periods (20)"),
        ("results/grand_melee_short_periods/results.csv", "Short Periods (3)"),
    ]

    melee_data = {}
    for path, name in scenarios:
        try:
            df = pd.read_csv(path)
            melee_data[name] = df
            print(f"✓ Loaded {name}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")

    return melee_data


def analyze_self_play():
    """Analyze pure self-play results."""
    print("\n" + "="*80)
    print("ANALYZING SELF-PLAY PERFORMANCE")
    print("="*80)

    # Try to load the most complete pure results
    pure_paths = [
        "results/tournament_all_v_all_20251124_042450/pure/",
        "results/tournament_pure_20251124_032925/pure/"
    ]

    results = []
    for base_path in pure_paths:
        pattern = f"{base_path}pure_*.csv"
        files = glob(pattern)
        if files:
            print(f"\nFound {len(files)} pure market files in {base_path}")
            for file in files:
                trader = Path(file).stem.replace('pure_', '')
                try:
                    df = pd.read_csv(file)
                    eff_mean = df['efficiency'].mean()
                    eff_std = df['efficiency'].std()
                    results.append({
                        'Trader': TRADER_NAMES.get(trader, trader),
                        'Efficiency_Mean': eff_mean,
                        'Efficiency_Std': eff_std,
                        'N_Periods': len(df)
                    })
                    print(f"  {trader}: {eff_mean:.1f}% ± {eff_std:.1f}%")
                except Exception as e:
                    print(f"  ✗ Error loading {trader}: {e}")
            break

    if not results:
        print("WARNING: No self-play data found!")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Efficiency_Mean', ascending=False)

    return df_results


def analyze_invasibility():
    """Analyze 1v7 invasibility results from tracker.md."""
    print("\n" + "="*80)
    print("ANALYZING INVASIBILITY PERFORMANCE")
    print("="*80)

    # Hardcode data from tracker.md since parsing is complex
    invasibility_data = [
        {'Trader': 'Perry', 'Overall': 2.10, 'Buyer': 0.75, 'Seller': 3.44},
        {'Trader': 'GD', 'Overall': 1.83, 'Buyer': 1.10, 'Seller': 2.57},
        {'Trader': 'Lin', 'Overall': 1.77, 'Buyer': 1.24, 'Seller': 2.30},
        {'Trader': 'ZIP', 'Overall': 1.25, 'Buyer': 0.15, 'Seller': 2.35},
        {'Trader': 'ZI2', 'Overall': 0.94, 'Buyer': 1.05, 'Seller': 0.83},
    ]

    df = pd.DataFrame(invasibility_data)
    df = df.sort_values('Overall', ascending=False)

    for _, row in df.iterrows():
        print(f"  {row['Trader']}: {row['Overall']:.2f}x (B:{row['Buyer']:.2f}x, S:{row['Seller']:.2f}x)")

    return df


def analyze_grand_melee(melee_data):
    """Analyze Grand Melee tournament results."""
    print("\n" + "="*80)
    print("ANALYZING GRAND MELEE TOURNAMENT")
    print("="*80)

    if not melee_data:
        print("WARNING: No melee data available!")
        return pd.DataFrame()

    # Calculate rankings per scenario
    rankings = []
    for scenario_name, df in melee_data.items():
        try:
            # Group by trader and calculate mean profit
            trader_profits = df.groupby('agent_type')['period_profit'].mean()
            trader_profits = trader_profits.sort_values(ascending=False)

            for rank, (trader, profit) in enumerate(trader_profits.items(), 1):
                rankings.append({
                    'Scenario': scenario_name,
                    'Trader': TRADER_NAMES.get(trader, trader),
                    'Rank': rank,
                    'Profit': profit
                })
            print(f"  {scenario_name}: #1 = {trader_profits.index[0]}")
        except Exception as e:
            print(f"  ✗ Error processing {scenario_name}: {e}")

    df_rankings = pd.DataFrame(rankings)

    # Calculate average rank per trader
    avg_ranks = df_rankings.groupby('Trader')['Rank'].mean().sort_values()
    print("\nAverage Ranks:")
    for trader, rank in avg_ranks.items():
        print(f"  {trader}: {rank:.1f}")

    return df_rankings


def generate_table_selfplay(df_selfplay):
    """Generate LaTeX table for self-play results."""
    print("\nGenerating Table 1: Self-Play Efficiency...")

    latex = r"""\begin{table}[t]
\centering
\caption{Pure Self-Play Performance (All traders of same type)}
\label{tab:selfplay}
\begin{tabular}{lrrr}
\toprule
\textbf{Trader} & \textbf{Efficiency (\%)} & \textbf{Std Dev} & \textbf{Status} \\
\midrule
"""

    for _, row in df_selfplay.iterrows():
        eff = row['Efficiency_Mean']
        std = row['Efficiency_Std']

        # Determine status
        if eff > 95:
            status = r"\cellcolor{green!20}Excellent"
        elif eff > 85:
            status = r"\cellcolor{yellow!20}Good"
        elif eff > 70:
            status = r"\cellcolor{orange!20}Moderate"
        else:
            status = r"\cellcolor{red!20}Poor"

        latex += f"{row['Trader']} & {eff:.1f} & {std:.1f} & {status} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = FIGURES_DIR / "table_selfplay.tex"
    output_path.write_text(latex)
    print(f"  Saved to {output_path}")


def generate_table_invasibility(df_inv):
    """Generate LaTeX table for invasibility results."""
    print("\nGenerating Table 2: Invasibility Performance...")

    latex = r"""\begin{table}[t]
\centering
\caption{Invasibility Test: 1 Trader vs 7 ZIC Agents (8v8 Market)}
\label{tab:invasibility}
\begin{tabular}{lrrr}
\toprule
\textbf{Trader} & \textbf{Overall Ratio} & \textbf{As Buyer} & \textbf{As Seller} \\
\midrule
"""

    for _, row in df_inv.iterrows():
        latex += f"{row['Trader']} & {row['Overall']:.2f}x & {row['Buyer']:.2f}x & {row['Seller']:.2f}x \\\\\n"

    latex += r"""\bottomrule
\multicolumn{4}{l}{\footnotesize Ratio = (Trader profit / agent) / (ZIC profit / agent)} \\
\multicolumn{4}{l}{\footnotesize Target: >3.0x indicates strong exploitation capability} \\
\end{tabular}
\end{table}
"""

    output_path = FIGURES_DIR / "table_invasibility.tex"
    output_path.write_text(latex)
    print(f"  Saved to {output_path}")


def generate_table_melee(df_rankings):
    """Generate LaTeX table for Grand Melee results."""
    print("\nGenerating Table 3: Grand Melee Tournament...")

    # Calculate statistics per trader
    trader_stats = df_rankings.groupby('Trader').agg({
        'Rank': ['mean', 'std', 'min', 'max']
    }).round(1)
    trader_stats.columns = ['Avg_Rank', 'Std_Rank', 'Best_Rank', 'Worst_Rank']
    trader_stats = trader_stats.sort_values('Avg_Rank')

    latex = r"""\begin{table}[t]
\centering
\caption{Grand Melee Tournament: Average Rank Across 10 Market Scenarios}
\label{tab:melee}
\begin{tabular}{lrrrr}
\toprule
\textbf{Trader} & \textbf{Avg Rank} & \textbf{Std Dev} & \textbf{Best} & \textbf{Worst} \\
\midrule
"""

    for trader, row in trader_stats.iterrows():
        latex += f"{trader} & {row['Avg_Rank']:.1f} & {row['Std_Rank']:.1f} & {int(row['Best_Rank'])} & {int(row['Worst_Rank'])} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{5}{l}{\footnotesize Lower rank = better performance. Scenarios include baseline,} \\
\multicolumn{5}{l}{\footnotesize asymmetric markets, time pressure, and token scarcity.} \\
\end{tabular}
\end{table}
"""

    output_path = FIGURES_DIR / "table_melee.tex"
    output_path.write_text(latex)
    print(f"  Saved to {output_path}")


def generate_figure_selfplay(df_selfplay):
    """Generate bar chart for self-play efficiency."""
    print("\nGenerating Figure 1: Self-Play Bar Chart...")

    if df_selfplay.empty:
        print("  WARNING: No data, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    traders = df_selfplay['Trader'].values
    efficiencies = df_selfplay['Efficiency_Mean'].values
    stds = df_selfplay['Efficiency_Std'].values

    colors = [TRADER_COLORS.get(t, '#95a5a6') for t in traders]

    bars = ax.barh(traders, efficiencies, xerr=stds, color=colors, alpha=0.8, capsize=5)

    # Add reference lines
    ax.axvline(85, color='green', linestyle='--', alpha=0.5, label='Target (85%)')
    ax.axvline(98, color='blue', linestyle='--', alpha=0.5, label='Excellent (98%)')

    ax.set_xlabel('Efficiency (%)', fontsize=12)
    ax.set_ylabel('Trader', fontsize=12)
    ax.set_title('Pure Self-Play Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_selfplay_bar.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def generate_figure_invasibility(df_inv):
    """Generate heatmap for invasibility results."""
    print("\nGenerating Figure 2: Invasibility Heatmap...")

    if df_inv.empty:
        print("  WARNING: No data, skipping figure")
        return

    # Prepare data for heatmap
    traders = df_inv['Trader'].values
    data = df_inv[['Buyer', 'Seller']].values

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['As Buyer', 'As Seller'])
    ax.set_yticks(range(len(traders)))
    ax.set_yticklabels(traders)

    # Add text annotations
    for i in range(len(traders)):
        for j in range(2):
            text = ax.text(j, i, f'{data[i, j]:.2f}x',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Invasibility: Profit Ratio vs 7 ZIC Agents', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Profit Ratio vs ZIC', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_invasibility_heatmap.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def generate_figure_melee_heatmap(df_rankings):
    """Generate heatmap for Grand Melee rankings."""
    print("\nGenerating Figure 3: Grand Melee Heatmap...")

    if df_rankings.empty:
        print("  WARNING: No data, skipping figure")
        return

    # Pivot to create trader x scenario matrix
    pivot = df_rankings.pivot(index='Trader', columns='Scenario', values='Rank')

    # Sort by average rank
    pivot['Avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Avg')
    pivot = pivot.drop('Avg', axis=1)

    # Shorten scenario names for display
    pivot.columns = [c.split('(')[0].strip() for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=9)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{int(val)}',
                              ha="center", va="center", color="white", fontsize=8,
                              fontweight='bold')

    ax.set_title('Grand Melee: Rank by Scenario', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank (1=Best)', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_melee_heatmap.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def generate_figure_efficiency_paradox(df_selfplay, df_rankings):
    """Generate scatter plot showing self-play efficiency vs tournament rank."""
    print("\nGenerating Figure 4: Efficiency Paradox Scatter...")

    if df_selfplay.empty or df_rankings.empty:
        print("  WARNING: Insufficient data, skipping figure")
        return

    # Calculate average tournament rank
    avg_ranks = df_rankings.groupby('Trader')['Rank'].mean()

    # Merge with self-play data
    plot_data = []
    for _, row in df_selfplay.iterrows():
        trader = row['Trader']
        if trader in avg_ranks.index:
            plot_data.append({
                'Trader': trader,
                'Efficiency': row['Efficiency_Mean'],
                'Rank': avg_ranks[trader]
            })

    if not plot_data:
        print("  WARNING: No matching data, skipping figure")
        return

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot points
    for _, row in df_plot.iterrows():
        trader = row['Trader']
        ax.scatter(row['Efficiency'], row['Rank'],
                  s=150, color=TRADER_COLORS.get(trader, '#95a5a6'),
                  alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.text(row['Efficiency']+1, row['Rank'], trader,
               fontsize=9, va='center')

    ax.set_xlabel('Self-Play Efficiency (%)', fontsize=12)
    ax.set_ylabel('Average Tournament Rank', fontsize=12)
    ax.set_title('The Self-Play Efficiency Paradox', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Lower rank = better
    ax.grid(True, alpha=0.3)

    # Add annotation box
    textstr = 'High cooperation efficiency\n≠ Competitive success'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_efficiency_paradox.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def save_summary_stats(df_selfplay, df_inv, df_rankings):
    """Save summary statistics to JSON."""
    print("\nSaving summary statistics...")

    summary = {
        'self_play': {
            'n_traders': len(df_selfplay),
            'mean_efficiency': float(df_selfplay['Efficiency_Mean'].mean()) if not df_selfplay.empty else 0,
            'best_trader': df_selfplay.iloc[0]['Trader'] if not df_selfplay.empty else None,
            'best_efficiency': float(df_selfplay.iloc[0]['Efficiency_Mean']) if not df_selfplay.empty else 0
        },
        'invasibility': {
            'n_traders': len(df_inv),
            'strongest_invader': df_inv.iloc[0]['Trader'] if not df_inv.empty else None,
            'strongest_ratio': float(df_inv.iloc[0]['Overall']) if not df_inv.empty else 0
        },
        'grand_melee': {
            'n_scenarios': len(df_rankings['Scenario'].unique()) if not df_rankings.empty else 0,
            'n_traders': len(df_rankings['Trader'].unique()) if not df_rankings.empty else 0,
            'champion': df_rankings.groupby('Trader')['Rank'].mean().idxmin() if not df_rankings.empty else None
        }
    }

    output_path = FIGURES_DIR / "summary_stats.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to {output_path}")
    print("\nSummary:")
    print(f"  Best Self-Play: {summary['self_play']['best_trader']} ({summary['self_play']['best_efficiency']:.1f}%)")
    print(f"  Strongest Invader: {summary['invasibility']['strongest_invader']} ({summary['invasibility']['strongest_ratio']:.2f}x)")
    print(f"  Melee Champion: {summary['grand_melee']['champion']}")


def main():
    print("="*80)
    print("LEGACY TRADER PERFORMANCE ANALYSIS")
    print("Generating tables and figures for paper section 04a")
    print("="*80)

    # Load data
    df_selfplay = analyze_self_play()
    df_inv = analyze_invasibility()
    melee_data = load_grand_melee_data()
    df_rankings = analyze_grand_melee(melee_data)

    # Generate tables
    print("\n" + "="*80)
    print("GENERATING LATEX TABLES")
    print("="*80)
    if not df_selfplay.empty:
        generate_table_selfplay(df_selfplay)
    if not df_inv.empty:
        generate_table_invasibility(df_inv)
    if not df_rankings.empty:
        generate_table_melee(df_rankings)

    # Generate figures
    print("\n" + "="*80)
    print("GENERATING PDF FIGURES")
    print("="*80)
    generate_figure_selfplay(df_selfplay)
    generate_figure_invasibility(df_inv)
    generate_figure_melee_heatmap(df_rankings)
    generate_figure_efficiency_paradox(df_selfplay, df_rankings)

    # Save summary
    save_summary_stats(df_selfplay, df_inv, df_rankings)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {FIGURES_DIR}/")
    print("  - 3 LaTeX tables (table_*.tex)")
    print("  - 4 PDF figures (fig_*.pdf)")
    print("  - 1 Summary JSON (summary_stats.json)")
    print("\nNext steps:")
    print("  1. Review generated files")
    print("  2. Create paper/arxiv/sections/04a_legacy_trader_results.tex")
    print("  3. Update paper/arxiv/main.tex to include new section")


if __name__ == "__main__":
    main()

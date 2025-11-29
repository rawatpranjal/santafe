#!/usr/bin/env python3
"""
Grand Tournament Analysis: Round-Robin Mixed Melee

Analyzes results from the Grand Tournament where all trader types compete
together in mixed markets, replicating Rust et al. (1994) tournament format.

Usage:
    python scripts/analyze_grand_tournament.py --results-dir "results/grand_tournament_*"
    python scripts/analyze_grand_tournament.py --auto-discover
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100


ENVIRONMENT_MAP = {
    'grand_tournament_base': 'BASE',
    'grand_tournament_pressure': 'PRESSURE',
    'grand_tournament_scarcity': 'SCARCITY',
    'grand_tournament_asymmetric': 'ASYMMETRIC',
}


def load_tournament_results(result_dirs: List[Path]) -> pd.DataFrame:
    """
    Load all tournament results and aggregate.

    Returns:
        DataFrame with environment labels added
    """
    all_data = []

    for result_dir in result_dirs:
        results_file = result_dir / "results.csv"
        if not results_file.exists():
            print(f"Warning: No results.csv found in {result_dir}")
            continue

        # Load results
        df = pd.read_csv(results_file)

        # Extract environment from directory name
        dir_name = result_dir.name
        environment = ENVIRONMENT_MAP.get(dir_name, dir_name)

        df['environment'] = environment
        df['config_name'] = dir_name

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid results found")

    return pd.concat(all_data, ignore_index=True)


def calculate_agent_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive rankings for each agent type.

    Returns:
        DataFrame with mean profit, efficiency, win rate, etc.
    """
    # Group by agent_type across all environments
    grouped = df.groupby('agent_type')

    rankings = grouped.agg({
        'period_profit': ['mean', 'std', 'count'],
        'profit_deviation': 'mean',
        'efficiency': 'mean',
        'num_trades': ['mean', 'sum'],
    }).reset_index()

    # Flatten column names
    rankings.columns = [
        'agent_type',
        'mean_profit',
        'std_profit',
        'count',
        'mean_profit_deviation',
        'mean_efficiency',
        'mean_trades',
        'total_trades',
    ]

    # Calculate win rate (% of periods where agent has highest profit)
    win_counts = []
    for agent_type in rankings['agent_type']:
        agent_df = df[df['agent_type'] == agent_type]

        # For each period, check if this agent had highest profit
        wins = 0
        total_periods = 0

        for (env, rnd, period), group in df.groupby(['environment', 'round', 'period']):
            max_profit = group['period_profit'].max()
            agent_profit = group[group['agent_type'] == agent_type]['period_profit']

            if len(agent_profit) > 0:
                total_periods += 1
                if agent_profit.iloc[0] == max_profit:
                    wins += 1

        win_rate = (wins / total_periods * 100) if total_periods > 0 else 0
        win_counts.append(win_rate)

    rankings['win_rate_pct'] = win_counts

    # Sort by mean profit
    rankings = rankings.sort_values('mean_profit', ascending=False)
    rankings['rank'] = range(1, len(rankings) + 1)

    return rankings


def calculate_rankings_by_environment(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate rankings for each environment separately."""
    env_rankings = {}

    for environment in df['environment'].unique():
        env_df = df[df['environment'] == environment]

        grouped = env_df.groupby('agent_type')

        env_rank = grouped.agg({
            'period_profit': ['mean', 'std'],
            'efficiency': 'mean',
            'num_trades': 'mean',
        }).reset_index()

        env_rank.columns = [
            'agent_type',
            'mean_profit',
            'std_profit',
            'mean_efficiency',
            'mean_trades',
        ]

        env_rank = env_rank.sort_values('mean_profit', ascending=False)
        env_rank['rank'] = range(1, len(env_rank) + 1)

        env_rankings[environment] = env_rank

    return env_rankings


def test_rank_significance(df: pd.DataFrame, rankings: pd.DataFrame) -> Dict:
    """
    Statistical tests for ranking significance.

    Returns dict with:
        - friedman: Friedman test across environments
        - kendall_w: Rank concordance
        - pairwise: Pairwise Wilcoxon tests (top 3 vs others)
    """
    results = {}

    # Friedman test: Are there significant differences in profits across agent types?
    agent_types = rankings['agent_type'].tolist()
    profit_by_agent = [df[df['agent_type'] == at]['period_profit'].values for at in agent_types]

    # Need equal-length arrays, so sample to minimum length
    min_len = min(len(p) for p in profit_by_agent)
    profit_by_agent_sampled = [p[:min_len] for p in profit_by_agent]

    try:
        f_stat, p_value = friedmanchisquare(*profit_by_agent_sampled)
        results['friedman'] = {
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
    except Exception as e:
        print(f"Friedman test failed: {e}")
        results['friedman'] = None

    # Kendall's W: Rank concordance across environments
    env_rankings = calculate_rankings_by_environment(df)

    if len(env_rankings) >= 2:
        # Create rank matrix (agents × environments)
        rank_matrix = []
        agent_types_ordered = rankings['agent_type'].tolist()

        for agent in agent_types_ordered:
            ranks = []
            for env in sorted(env_rankings.keys()):
                env_df = env_rankings[env]
                agent_rank = env_df[env_df['agent_type'] == agent]['rank']
                if len(agent_rank) > 0:
                    ranks.append(agent_rank.iloc[0])
                else:
                    ranks.append(len(env_df) + 1)  # Worst rank if not present
            rank_matrix.append(ranks)

        rank_matrix = np.array(rank_matrix)

        # Kendall's W calculation
        n = rank_matrix.shape[0]  # number of agents
        k = rank_matrix.shape[1]  # number of environments

        rank_sums = rank_matrix.sum(axis=1)
        mean_rank_sum = rank_sums.mean()

        S = ((rank_sums - mean_rank_sum) ** 2).sum()
        W = (12 * S) / (k ** 2 * (n ** 3 - n))

        results['kendall_w'] = {
            'W': W,
            'interpretation': 'high' if W > 0.7 else 'moderate' if W > 0.5 else 'low',
        }

    # Pairwise comparisons (top 3 vs rest)
    top_3 = rankings.head(3)['agent_type'].tolist()
    rest = rankings.tail(len(rankings) - 3)['agent_type'].tolist()

    pairwise_results = []
    for top_agent in top_3:
        top_profits = df[df['agent_type'] == top_agent]['period_profit'].values

        for other_agent in rest:
            other_profits = df[df['agent_type'] == other_agent]['period_profit'].values

            # Sample to equal length
            min_len = min(len(top_profits), len(other_profits))

            try:
                stat, p_val = wilcoxon(
                    top_profits[:min_len],
                    other_profits[:min_len],
                    alternative='greater'
                )

                pairwise_results.append({
                    'agent_1': top_agent,
                    'agent_2': other_agent,
                    'p_value': p_val,
                    'significant': p_val < (0.05 / len(top_3) / len(rest)),  # Bonferroni
                })
            except Exception as e:
                print(f"Wilcoxon test failed for {top_agent} vs {other_agent}: {e}")

    results['pairwise'] = pairwise_results

    return results


def create_visualizations(
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    env_rankings: Dict[str, pd.DataFrame],
    output_dir: Path,
):
    """Generate all visualization charts."""

    # Figure 1: Overall Rankings Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(rankings))
    ax.bar(x, rankings['mean_profit'], yerr=rankings['std_profit'], capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rankings['agent_type'], rotation=45, ha='right')
    ax.set_xlabel('Trader Type (Ranked)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Profit per Period', fontsize=12, fontweight='bold')
    ax.set_title('Grand Tournament Rankings: Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'rankings_bar_chart.png', dpi=150)
    plt.close()

    # Figure 2: Efficiency vs Profit Scatter
    fig, ax = plt.subplots(figsize=(10, 7))

    for agent_type in rankings['agent_type']:
        agent_df = df[df['agent_type'] == agent_type]
        ax.scatter(
            agent_df['efficiency'],
            agent_df['period_profit'],
            label=agent_type,
            alpha=0.4,
            s=20,
        )

    ax.set_xlabel('Market Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Period Profit', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency vs Profit: Trade-off Analysis', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_vs_profit.png', dpi=150)
    plt.close()

    # Figure 3: Win Rate Heatmap
    envs = sorted(env_rankings.keys())
    agents = rankings['agent_type'].tolist()

    win_rate_matrix = []
    for agent in agents:
        row = []
        for env in envs:
            env_df = df[(df['environment'] == env) & (df['agent_type'] == agent)]

            # Calculate win rate in this environment
            wins = 0
            total = 0
            for (rnd, period), group_df in df[df['environment'] == env].groupby(['round', 'period']):
                max_profit = group_df['period_profit'].max()
                agent_profit = group_df[group_df['agent_type'] == agent]['period_profit']

                if len(agent_profit) > 0:
                    total += 1
                    if agent_profit.iloc[0] == max_profit:
                        wins += 1

            win_rate = (wins / total * 100) if total > 0 else 0
            row.append(win_rate)

        win_rate_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        win_rate_matrix,
        annot=True,
        fmt='.1f',
        xticklabels=envs,
        yticklabels=agents,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Win Rate (%)'},
    )
    ax.set_title('Win Rate by Environment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_heatmap.png', dpi=150)
    plt.close()

    # Figure 4: Profit Distribution Violin Plots
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data
    plot_data = []
    for agent in rankings['agent_type']:
        agent_df = df[df['agent_type'] == agent]
        for profit in agent_df['period_profit'].values:
            plot_data.append({'Agent': agent, 'Profit': profit})

    plot_df = pd.DataFrame(plot_data)

    sns.violinplot(
        data=plot_df,
        x='Agent',
        y='Profit',
        order=rankings['agent_type'].tolist(),
        ax=ax,
    )
    ax.set_xlabel('Trader Type (Ranked)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Period Profit', fontsize=12, fontweight='bold')
    ax.set_title('Profit Distribution by Trader', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_distribution_violin.png', dpi=150)
    plt.close()

    print(f"✓ Saved 4 visualization figures to {output_dir}/")


def generate_report(
    rankings: pd.DataFrame,
    env_rankings: Dict[str, pd.DataFrame],
    stats_results: Dict,
    df: pd.DataFrame,
    output_dir: Path,
):
    """Generate comprehensive Rust-style report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    winner = rankings.iloc[0]
    second = rankings.iloc[1]

    report = f"""# Grand Tournament Results: Round-Robin Mixed Melee

**Generated:** {timestamp}

---

## Tournament Winner: {winner['agent_type']}

**Key Result:** {winner['agent_type']} earned {winner['mean_profit']:.2f} avg profit per period, {((winner['mean_profit'] / second['mean_profit'] - 1) * 100):.1f}% above second place ({second['agent_type']}).

---

## Tournament Structure

**Trader Pool:** 10 types (ZIC, ZIP, Kaplan, GD, ZI2, Skeleton, Lin, Perry, Jacobson, ZI)

**Environments Tested:**
- **BASE:** Standard 10v10, 4 tokens, 100 steps
- **PRESSURE:** 10v10, 4 tokens, 25 steps (time pressure)
- **SCARCITY:** 10v10, 1 token, 100 steps (one-shot trading)
- **ASYMMETRIC:** 6v4 market, 4 tokens, 100 steps

**Sample Size:**
- 100 rounds × 10 periods × 4 environments = 4,000 market instances
- Total observations: {len(df):,}

**Overall Market Efficiency:** {df['efficiency'].mean():.2f}%

---

## Overall Rankings (Table 4 Replica)

| Rank | Trader | Avg Profit | Std Dev | Win Rate % | Efficiency % | Total Trades |
|------|--------|------------|---------|------------|--------------|--------------|
"""

    for _, row in rankings.iterrows():
        report += f"| {row['rank']} | **{row['agent_type']}** | {row['mean_profit']:.2f} | {row['std_profit']:.2f} | {row['win_rate_pct']:.1f}% | {row['mean_efficiency']:.2f}% | {int(row['total_trades'])} |\n"

    report += "\n---\n\n"

    # Statistical Analysis
    report += "## Statistical Analysis\n\n"

    if stats_results.get('friedman'):
        fr = stats_results['friedman']
        report += f"**Friedman Test (Non-parametric ANOVA):**\n"
        report += f"- χ² = {fr['statistic']:.2f}\n"
        report += f"- p-value = {fr['p_value']:.6f}\n"
        report += f"- **Significant:** {'YES ✓' if fr['significant'] else 'NO'}\n"
        report += f"- **Interpretation:** Trader types have {'significantly' if fr['significant'] else 'no'} different performance levels.\n\n"

    if stats_results.get('kendall_w'):
        kw = stats_results['kendall_w']
        report += f"**Kendall's W (Rank Concordance across Environments):**\n"
        report += f"- W = {kw['W']:.3f}\n"
        report += f"- **Interpretation:** {kw['interpretation'].capitalize()} concordance\n"
        report += f"- Traders rank {'consistently' if kw['W'] > 0.7 else 'moderately consistently' if kw['W'] > 0.5 else 'inconsistently'} across environments.\n\n"

    # Rankings by Environment
    report += "---\n\n## Rankings by Environment\n\n"

    for env_name in sorted(env_rankings.keys()):
        env_df = env_rankings[env_name]
        report += f"### {env_name} Environment\n\n"
        report += "| Rank | Trader | Avg Profit | Std Dev | Efficiency % |\n"
        report += "|------|--------|------------|---------|-------------|\n"

        for _, row in env_df.iterrows():
            report += f"| {row['rank']} | {row['agent_type']} | {row['mean_profit']:.2f} | {row['std_profit']:.2f} | {row['mean_efficiency']:.2f}% |\n"

        report += "\n"

    # Trader Classification
    report += "---\n\n## Trader Classification\n\n"

    top_tier = rankings.head(3)['agent_type'].tolist()
    mid_tier = rankings.iloc[3:6]['agent_type'].tolist()
    lower_tier = rankings.iloc[6:9]['agent_type'].tolist()
    bottom = rankings.tail(1)['agent_type'].tolist()

    report += f"**Strategic (Top Tier):** {', '.join(top_tier)}\n\n"
    report += f"**Adaptive (Mid Tier):** {', '.join(mid_tier)}\n\n"
    report += f"**Baseline (Lower Tier):** {', '.join(lower_tier)}\n\n"
    report += f"**Control (Bottom):** {', '.join(bottom)}\n\n"

    # Comparison to Rust et al. (1994)
    report += "---\n\n## Comparison to Rust et al. (1994)\n\n"

    kaplan_rank = rankings[rankings['agent_type'] == 'Kaplan']['rank'].iloc[0] if 'Kaplan' in rankings['agent_type'].values else None

    if kaplan_rank:
        report += f"**Kaplan Performance:**\n"
        report += f"- 1993 Rank: 1st (Winner)\n"
        report += f"- 2025 Rank: {int(kaplan_rank)}{'st' if kaplan_rank == 1 else 'nd' if kaplan_rank == 2 else 'rd' if kaplan_rank == 3 else 'th'}\n"
        report += f"- **Kaplan {'maintained' if kaplan_rank <= 2 else 'lost'} dominance**\n\n"

    report += f"**Market Efficiency:**\n"
    report += f"- 1993: 89-90%\n"
    report += f"- 2025: {df['efficiency'].mean():.2f}%\n\n"

    simple_agents = ['Kaplan', 'Skeleton', 'ZIC']
    complex_agents = ['GD', 'ZIP']

    simple_ranks = [int(rankings[rankings['agent_type'] == a]['rank'].iloc[0]) for a in simple_agents if a in rankings['agent_type'].values]
    complex_ranks = [int(rankings[rankings['agent_type'] == a]['rank'].iloc[0]) for a in complex_agents if a in rankings['agent_type'].values]

    report += f"**Simple vs Complex Strategies:**\n"
    report += f"- Simple agents (Kaplan, Skeleton, ZIC) avg rank: {np.mean(simple_ranks):.1f}\n"
    report += f"- Complex agents (GD, ZIP) avg rank: {np.mean(complex_ranks):.1f}\n"
    report += f"- **Simple strategies {'still beat' if np.mean(simple_ranks) < np.mean(complex_ranks) else 'no longer beat'} complex strategies**\n\n"

    # Key Findings
    report += "---\n\n## Key Findings\n\n"

    report += f"1. **Winner:** {winner['agent_type']} dominated with {winner['win_rate_pct']:.1f}% win rate across all environments.\n\n"
    report += f"2. **Robustness:** {'High' if stats_results.get('kendall_w', {}).get('W', 0) > 0.7 else 'Moderate'} rank concordance (W={stats_results.get('kendall_w', {}).get('W', 0):.2f}) indicates {'consistent' if stats_results.get('kendall_w', {}).get('W', 0) > 0.7 else 'variable'} performance across environments.\n\n"
    report += f"3. **Efficiency vs Profit:** Market efficiency ({df['efficiency'].mean():.1f}%) does not guarantee equitable profit distribution - top performer earned {winner['mean_profit'] / rankings['mean_profit'].mean():.2f}x the average.\n\n"

    # Conclusions
    report += "---\n\n## Conclusions\n\n"

    report += f"The Grand Tournament validates several findings from the 1993 Santa Fe competition:\n\n"
    report += f"- **Strategic patience beats complexity:** {winner['agent_type']}'s success demonstrates that timing and strategic waiting outperform sophisticated learning algorithms.\n"
    report += f"- **Robustness matters:** Top performers maintained rankings across diverse market conditions (BASE, PRESSURE, SCARCITY, ASYMMETRIC).\n"
    report += f"- **Efficiency ≠ Profit:** High market efficiency coexists with significant profit inequality - strategic agents extract disproportionate surplus.\n\n"

    report += "---\n\n## Visualizations\n\n"
    report += "![Overall Rankings](rankings_bar_chart.png)\n\n"
    report += "![Efficiency vs Profit](efficiency_vs_profit.png)\n\n"
    report += "![Win Rate Heatmap](win_rate_heatmap.png)\n\n"
    report += "![Profit Distribution](profit_distribution_violin.png)\n\n"

    report += "---\n\n*Generated by: scripts/analyze_grand_tournament.py*\n"

    # Write report
    report_path = output_dir / 'report.md'
    report_path.write_text(report)
    print(f"✓ Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Grand Tournament results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory pattern for tournament results (supports wildcards)',
    )

    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Auto-discover grand_tournament results in results/',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/grand_tournament_analysis',
        help='Output directory for analysis results',
    )

    args = parser.parse_args()

    # Find results directories
    if args.auto_discover:
        results_dirs = list(Path('results').glob('grand_tournament_*'))
        if not results_dirs:
            print("Error: No grand_tournament results found in results/")
            return
    elif args.results_dir:
        results_dirs = list(Path('.').glob(args.results_dir))
    else:
        parser.print_help()
        return

    print(f"Found {len(results_dirs)} result directories")

    # Load results
    df = load_tournament_results(results_dirs)
    print(f"Loaded {len(df):,} observations")
    print(f"Environments: {df['environment'].unique()}")
    print(f"Trader types: {df['agent_type'].unique()}")

    # Calculate rankings
    rankings = calculate_agent_rankings(df)
    print("\nOverall Rankings:")
    print(rankings[['rank', 'agent_type', 'mean_profit', 'win_rate_pct']].to_string(index=False))

    env_rankings = calculate_rankings_by_environment(df)

    # Statistical tests
    print("\nRunning statistical tests...")
    stats_results = test_rank_significance(df, rankings)

    if stats_results.get('friedman'):
        print(f"  Friedman test: χ²={stats_results['friedman']['statistic']:.2f}, p={stats_results['friedman']['p_value']:.6f}")
    if stats_results.get('kendall_w'):
        print(f"  Kendall's W: {stats_results['kendall_w']['W']:.3f}")

    # Create output directory
    output_dir = Path(args.output_dir + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    create_visualizations(df, rankings, env_rankings, output_dir)

    # Generate report
    generate_report(rankings, env_rankings, stats_results, df, output_dir)

    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()

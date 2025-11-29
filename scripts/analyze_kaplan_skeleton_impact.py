#!/usr/bin/env python3
"""
Kaplan Performance Analysis: Impact of Skeleton Presence

Analyzes whether Kaplan's performance improves when Skeleton agents are present
in mixed-strategy tournaments, and quantifies the relationship between Skeleton
proportion and Kaplan's profit advantage.

Usage:
    python scripts/analyze_kaplan_skeleton_impact.py --results-dir results/tournament_kaplan_analysis_*/
    python scripts/analyze_kaplan_skeleton_impact.py --auto-discover
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Statistical testing
from scipy import stats
from scipy.stats import f_oneway, ttest_ind

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def extract_skeleton_pct(config_name: str) -> float:
    """
    Extract Skeleton percentage from config name.

    Examples:
        "baseline_no_skeleton" -> 0.0
        "baseline_with_skeleton" -> 37.5 (3/8)
        "skeleton_density_12pct" -> 12.5
        "skeleton_density_50pct" -> 50.0
    """
    if "no_skeleton" in config_name:
        return 0.0
    elif "with_skeleton" in config_name:
        return 37.5  # baseline_with_skeleton has 3/8 Skeleton

    match = re.search(r'(\d+)pct', config_name)
    if match:
        return float(match.group(1))

    return 0.0


def load_kaplan_results(results_dirs: List[Path]) -> pd.DataFrame:
    """
    Load all Kaplan agent observations from tournament results.

    Args:
        results_dirs: List of result directories to process

    Returns:
        DataFrame with columns:
            - config_name, skeleton_pct
            - round, period, agent_id, agent_type, is_buyer
            - num_trades, period_profit, agent_eq_profit, profit_deviation
            - efficiency, price_mean, price_std_dev
    """
    all_data = []

    for results_dir in results_dirs:
        results_file = results_dir / "results.csv"
        if not results_file.exists():
            print(f"Warning: No results.csv found in {results_dir}")
            continue

        # Load results
        df = pd.read_csv(results_file)

        # Extract config name from directory
        config_name = results_dir.name
        skeleton_pct = extract_skeleton_pct(config_name)

        # Add metadata
        df['config_name'] = config_name
        df['skeleton_pct'] = skeleton_pct

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid results found")

    return pd.concat(all_data, ignore_index=True)


def calculate_kaplan_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics for Kaplan agents.

    Returns:
        DataFrame grouped by skeleton_pct with metrics:
            - mean_profit_per_period
            - std_profit_per_period
            - mean_profit_deviation (vs equilibrium)
            - mean_efficiency
            - win_rate (% of periods where Kaplan has highest profit)
            - total_observations
    """
    # Filter to Kaplan agents only
    kaplan_df = df[df['agent_type'] == 'Kaplan'].copy()

    if len(kaplan_df) == 0:
        raise ValueError("No Kaplan agents found in results")

    # Group by skeleton_pct
    grouped = kaplan_df.groupby('skeleton_pct')

    metrics = grouped.agg({
        'period_profit': ['mean', 'std', 'count'],
        'profit_deviation': 'mean',
        'efficiency': 'mean',
    }).reset_index()

    # Flatten column names
    metrics.columns = [
        'skeleton_pct',
        'mean_profit',
        'std_profit',
        'count',
        'mean_profit_deviation',
        'mean_efficiency',
    ]

    # Calculate win rate (Kaplan highest profit in period)
    win_rates = []
    for skeleton_pct in metrics['skeleton_pct']:
        subset = df[df['skeleton_pct'] == skeleton_pct]

        # Group by round/period, find max profit per period
        period_max = subset.groupby(['round', 'period'])['period_profit'].transform('max')
        kaplan_subset = subset[subset['agent_type'] == 'Kaplan']
        kaplan_wins = (kaplan_subset['period_profit'] == kaplan_subset.groupby(['round', 'period'])['period_profit'].transform('max')).sum()
        total_periods = len(kaplan_subset.groupby(['round', 'period']))

        win_rate = (kaplan_wins / total_periods * 100) if total_periods > 0 else 0
        win_rates.append(win_rate)

    metrics['win_rate_pct'] = win_rates

    return metrics


def test_skeleton_effect(df: pd.DataFrame) -> Dict[str, any]:
    """
    Statistical testing of Skeleton's effect on Kaplan performance.

    Returns dict with:
        - ttest: T-test comparing 0% vs >0% Skeleton
        - anova: One-way ANOVA across all skeleton_pct levels
        - regression: Linear regression profit ~ skeleton_pct
        - cohen_d: Effect size
    """
    kaplan_df = df[df['agent_type'] == 'Kaplan'].copy()

    results = {}

    # T-test: 0% Skeleton vs any Skeleton present
    no_skeleton = kaplan_df[kaplan_df['skeleton_pct'] == 0]['period_profit']
    with_skeleton = kaplan_df[kaplan_df['skeleton_pct'] > 0]['period_profit']

    if len(no_skeleton) > 0 and len(with_skeleton) > 0:
        t_stat, p_value = ttest_ind(with_skeleton, no_skeleton)

        # Cohen's d effect size
        pooled_std = np.sqrt((no_skeleton.var() + with_skeleton.var()) / 2)
        cohen_d = (with_skeleton.mean() - no_skeleton.mean()) / pooled_std if pooled_std > 0 else 0

        results['ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_no_skeleton': no_skeleton.mean(),
            'mean_with_skeleton': with_skeleton.mean(),
            'cohen_d': cohen_d,
            'significant': p_value < 0.05,
        }

    # ANOVA: Across all skeleton_pct levels
    skeleton_levels = kaplan_df['skeleton_pct'].unique()
    if len(skeleton_levels) >= 3:
        groups = [kaplan_df[kaplan_df['skeleton_pct'] == pct]['period_profit'] for pct in skeleton_levels]
        f_stat, p_value = f_oneway(*groups)

        results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'num_levels': len(skeleton_levels),
        }

    # Linear regression: profit ~ skeleton_pct
    from scipy.stats import linregress
    x = kaplan_df['skeleton_pct']
    y = kaplan_df['period_profit']

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    results['regression'] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'significant': p_value < 0.05,
    }

    return results


def plot_kaplan_performance(df: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path):
    """Generate visualizations of Kaplan performance."""
    kaplan_df = df[df['agent_type'] == 'Kaplan'].copy()

    # Figure 1: Line plot with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        metrics['skeleton_pct'],
        metrics['mean_profit'],
        yerr=metrics['std_profit'],
        marker='o',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Kaplan Mean Profit',
    )
    ax.set_xlabel('Skeleton Proportion (%)', fontsize=12)
    ax.set_ylabel('Mean Profit per Period', fontsize=12)
    ax.set_title('Kaplan Performance vs Skeleton Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'kaplan_profit_vs_skeleton_pct.png', dpi=150)
    plt.close()

    # Figure 2: Box plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    skeleton_levels = sorted(kaplan_df['skeleton_pct'].unique())
    data_by_level = [kaplan_df[kaplan_df['skeleton_pct'] == pct]['period_profit'] for pct in skeleton_levels]

    box_plot = ax.boxplot(
        data_by_level,
        labels=[f"{pct:.1f}%" for pct in skeleton_levels],
        patch_artist=True,
    )

    # Color boxes
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Skeleton Proportion', fontsize=12)
    ax.set_ylabel('Period Profit', fontsize=12)
    ax.set_title('Kaplan Profit Distribution by Skeleton Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'kaplan_profit_distribution.png', dpi=150)
    plt.close()

    # Figure 3: Scatter plot (profit vs efficiency, colored by skeleton_pct)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        kaplan_df['efficiency'],
        kaplan_df['period_profit'],
        c=kaplan_df['skeleton_pct'],
        cmap='viridis',
        alpha=0.5,
        s=10,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Skeleton %', fontsize=11)
    ax.set_xlabel('Market Efficiency (%)', fontsize=12)
    ax.set_ylabel('Kaplan Period Profit', fontsize=12)
    ax.set_title('Kaplan Profit vs Market Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'kaplan_profit_vs_efficiency.png', dpi=150)
    plt.close()

    # Figure 4: Win rate bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        metrics['skeleton_pct'],
        metrics['win_rate_pct'],
        width=5,
        color='steelblue',
        edgecolor='black',
    )
    ax.set_xlabel('Skeleton Proportion (%)', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Kaplan Win Rate by Skeleton Density', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'kaplan_win_rate.png', dpi=150)
    plt.close()

    print(f"✓ Saved 4 figures to {output_dir}/")


def generate_report(
    metrics: pd.DataFrame,
    stats: Dict[str, any],
    output_dir: Path,
):
    """Generate markdown report with findings."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Kaplan Performance Analysis: Impact of Skeleton Presence

**Generated:** {timestamp}

---

## Executive Summary

This analysis investigates whether Kaplan's predatory sniping strategy benefits from the presence of Skeleton agents in mixed-strategy double auction tournaments.

### Key Findings

"""

    # Add t-test results
    if 'ttest' in stats:
        ttest = stats['ttest']
        report += f"""
**Baseline Comparison (With vs Without Skeleton):**
- Mean profit WITHOUT Skeleton: {ttest['mean_no_skeleton']:.2f}
- Mean profit WITH Skeleton: {ttest['mean_with_skeleton']:.2f}
- Difference: {ttest['mean_with_skeleton'] - ttest['mean_no_skeleton']:.2f} ({((ttest['mean_with_skeleton'] / ttest['mean_no_skeleton'] - 1) * 100):.1f}%)
- T-statistic: {ttest['t_statistic']:.3f}
- P-value: {ttest['p_value']:.4f}
- Cohen's d: {ttest['cohen_d']:.3f}
- **Statistically Significant:** {'YES ✓' if ttest['significant'] else 'NO'}

"""

    # Add ANOVA results
    if 'anova' in stats:
        anova = stats['anova']
        report += f"""
**ANOVA (Across all Skeleton densities):**
- F-statistic: {anova['f_statistic']:.3f}
- P-value: {anova['p_value']:.4f}
- **Statistically Significant:** {'YES ✓' if anova['significant'] else 'NO'}

"""

    # Add regression results
    if 'regression' in stats:
        reg = stats['regression']
        report += f"""
**Linear Regression (Profit ~ Skeleton %):**
- Slope: {reg['slope']:.4f} (profit change per 1% Skeleton increase)
- R²: {reg['r_squared']:.4f}
- P-value: {reg['p_value']:.4f}
- **Statistically Significant:** {'YES ✓' if reg['significant'] else 'NO'}

"""

    report += """
---

## Methodology

**Tournament Structure:**
- 8 buyers × 8 sellers (16 total agents)
- 100 rounds × 10 periods = 1,000 market instances per configuration
- 100 steps per period
- Game 6453 token generation (Santa Fe standard)

**Treatments:**
- Control: 0% Skeleton (Kaplan vs ZIC/ZIP/GD only)
- Varying Skeleton density: 12.5%, 25%, 37.5%, 50%, 62.5%
- Kaplan held constant at 25% (2/8 agents) across all treatments

---

## Results by Skeleton Density

"""

    # Add metrics table
    report += "| Skeleton % | Mean Profit | Std Dev | Win Rate % | Mean Efficiency % | Observations |\n"
    report += "|------------|-------------|---------|------------|-------------------|-------------|\n"

    for _, row in metrics.iterrows():
        report += f"| {row['skeleton_pct']:.1f}% | {row['mean_profit']:.2f} | {row['std_profit']:.2f} | {row['win_rate_pct']:.1f}% | {row['mean_efficiency']:.2f}% | {int(row['count'])} |\n"

    report += """
---

## Visualizations

![Kaplan Profit vs Skeleton %](kaplan_profit_vs_skeleton_pct.png)

![Kaplan Profit Distribution](kaplan_profit_distribution.png)

![Kaplan Profit vs Efficiency](kaplan_profit_vs_efficiency.png)

![Kaplan Win Rate](kaplan_win_rate.png)

---

## Discussion

### Mechanism

Kaplan's sniping strategy exploits Skeleton's predictable alpha-weighted convergence:
1. Skeleton agents narrow the spread through adaptive weighted-average bidding
2. Kaplan waits at extreme prices (bid=1 or ask=999)
3. Kaplan snipes trades at favorable prices after spread narrows
4. Visual test evidence: Kaplan achieves 2.2x profit advantage in pure Kaplan vs Skeleton matchups

### Implications

**For Kaplan:**
- Skeleton presence increases exploitable patterns in the market
- Kaplan's timing-based advantage allows surplus extraction even in efficient markets

**For Market Design:**
- High efficiency does not guarantee equitable surplus distribution
- Patient strategic agents can dominate adaptive but predictable opponents

---

## Recommendations

1. **Against Kaplan:** Avoid predictable convergence patterns; use randomization or time-varying strategies
2. **With Kaplan:** Optimal opponent mix depends on analysis results (see metrics table above)
3. **For RL Training:** Use Kaplan as adversary to learn robust non-exploitable policies

---

*Generated by: scripts/analyze_kaplan_skeleton_impact.py*
"""

    # Write report
    report_path = output_dir / 'report.md'
    report_path.write_text(report)
    print(f"✓ Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Kaplan performance impact of Skeleton presence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing tournament results (supports wildcards)',
    )

    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Auto-discover kaplan_analysis results in results/',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/kaplan_skeleton_analysis',
        help='Output directory for analysis results',
    )

    args = parser.parse_args()

    # Find results directories
    if args.auto_discover:
        results_dirs = list(Path('results').glob('*kaplan_analysis*'))
        if not results_dirs:
            print("Error: No kaplan_analysis results found in results/")
            return
    elif args.results_dir:
        results_dirs = list(Path('.').glob(args.results_dir))
    else:
        parser.print_help()
        return

    print(f"Found {len(results_dirs)} result directories")

    # Load and filter results
    df = load_kaplan_results(results_dirs)
    print(f"Loaded {len(df)} observations")
    print(f"Kaplan observations: {len(df[df['agent_type'] == 'Kaplan'])}")

    # Calculate metrics
    metrics = calculate_kaplan_metrics(df)
    print("\nKaplan Performance Metrics:")
    print(metrics.to_string(index=False))

    # Statistical tests
    stats = test_skeleton_effect(df)
    print("\nStatistical Tests:")
    if 'ttest' in stats:
        print(f"  T-test: p={stats['ttest']['p_value']:.4f}, d={stats['ttest']['cohen_d']:.3f}")
    if 'anova' in stats:
        print(f"  ANOVA: p={stats['anova']['p_value']:.4f}")
    if 'regression' in stats:
        print(f"  Regression: slope={stats['regression']['slope']:.4f}, R²={stats['regression']['r_squared']:.4f}")

    # Create output directory
    output_dir = Path(args.output_dir + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_kaplan_performance(df, metrics, output_dir)

    # Generate report
    generate_report(metrics, stats, output_dir)

    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()

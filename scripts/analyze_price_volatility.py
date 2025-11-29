#!/usr/bin/env python3
"""
Price Volatility Analyzer for Santa Fe Tournament.

Analyzes price volatility metrics across different trader types and market configurations.
Identifies stabilizing vs destabilizing traders based on price discovery patterns.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


class VolatilityAnalyzer:
    """Analyzes price volatility patterns across tournament experiments."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer with results directory.

        Args:
            results_dir: Tournament results directory with CSV files
        """
        self.results_dir = Path(results_dir)
        self.all_results = pd.DataFrame()
        self.traders = set()

    def load_results(self) -> pd.DataFrame:
        """Load all CSV results from tournament directory."""
        if not self.results_dir.exists():
            print(f"Error: {self.results_dir} does not exist")
            sys.exit(1)

        # Look for aggregate results first
        aggregate_file = self.results_dir / "aggregate_results.csv"
        if aggregate_file.exists():
            print(f"Loading aggregate results from {aggregate_file}")
            self.all_results = pd.read_csv(aggregate_file)
        else:
            # Load individual CSV files
            csv_files = list(self.results_dir.glob("*/results.csv"))
            if not csv_files:
                print(f"Error: No results.csv files found in {self.results_dir}")
                sys.exit(1)

            print(f"Found {len(csv_files)} CSV files in {self.results_dir}")
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['source_file'] = csv_file.parent.name
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

            self.all_results = pd.concat(dfs, ignore_index=True)

        # Verify volatility columns exist
        required_cols = ['price_std_dev', 'price_mean', 'price_volatility_pct']
        missing = [col for col in required_cols if col not in self.all_results.columns]
        if missing:
            print(f"Error: Missing volatility columns: {missing}")
            print(f"Available columns: {list(self.all_results.columns)}")
            sys.exit(1)

        # Track unique traders
        if 'agent_type' in self.all_results.columns:
            self.traders = set(self.all_results['agent_type'].unique())

        print(f"Loaded {len(self.all_results)} observations")
        print(f"Unique traders: {sorted(self.traders)}")

        return self.all_results

    def analyze_by_trader(self) -> pd.DataFrame:
        """Analyze volatility metrics grouped by trader type."""
        if self.all_results.empty:
            return pd.DataFrame()

        # Group by agent type and calculate statistics
        summary = self.all_results.groupby('agent_type').agg({
            'price_std_dev': ['mean', 'std', 'min', 'max'],
            'price_mean': ['mean'],
            'price_volatility_pct': ['mean', 'std', 'min', 'max'],
            'efficiency': ['mean']
        }).round(2)

        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        # Rename for clarity
        summary = summary.rename(columns={
            'price_std_dev_mean': 'avg_std_dev',
            'price_std_dev_std': 'std_of_std_dev',
            'price_std_dev_min': 'min_std_dev',
            'price_std_dev_max': 'max_std_dev',
            'price_mean_mean': 'avg_price',
            'price_volatility_pct_mean': 'avg_volatility_pct',
            'price_volatility_pct_std': 'std_volatility_pct',
            'price_volatility_pct_min': 'min_volatility_pct',
            'price_volatility_pct_max': 'max_volatility_pct',
            'efficiency_mean': 'avg_efficiency'
        })

        # Sort by average volatility (ascending = more stable)
        summary = summary.sort_values('avg_volatility_pct')

        print("\n" + "=" * 100)
        print("PRICE VOLATILITY BY TRADER TYPE")
        print("=" * 100)
        print(summary)
        print("\n")

        return summary

    def analyze_pure_vs_mixed(self) -> pd.DataFrame:
        """Compare volatility in pure (homogeneous) vs mixed markets."""
        if self.all_results.empty:
            return pd.DataFrame()

        # Try to infer market type from source file or configuration
        if 'source_file' in self.all_results.columns:
            self.all_results['market_type'] = self.all_results['source_file'].apply(
                lambda x: 'pure' if 'pure' in str(x).lower() else
                'mixed' if 'mixed' in str(x).lower() else 'other'
            )
        else:
            print("Warning: Cannot infer market type from data")
            return pd.DataFrame()

        comparison = self.all_results.groupby('market_type').agg({
            'price_volatility_pct': ['mean', 'std', 'count'],
            'efficiency': ['mean']
        }).round(2)

        comparison.columns = ['_'.join(col).strip('_') for col in comparison.columns.values]

        print("\n" + "=" * 80)
        print("PRICE VOLATILITY: PURE VS MIXED MARKETS")
        print("=" * 80)
        print(comparison)
        print("\n")

        return comparison

    def identify_stabilizers(self, threshold_pct: float = 10.0) -> Tuple[List[str], List[str]]:
        """
        Identify stabilizing vs destabilizing traders.

        Args:
            threshold_pct: Volatility threshold (default 10%)

        Returns:
            (stabilizers, destabilizers) tuple of trader lists
        """
        if self.all_results.empty:
            return ([], [])

        avg_volatility = self.all_results.groupby('agent_type')['price_volatility_pct'].mean()

        stabilizers = list(avg_volatility[avg_volatility <= threshold_pct].index)
        destabilizers = list(avg_volatility[avg_volatility > threshold_pct].index)

        print("\n" + "=" * 80)
        print(f"TRADER CLASSIFICATION (threshold: {threshold_pct}% volatility)")
        print("=" * 80)
        print(f"\nSTABILIZERS (volatility <= {threshold_pct}%):")
        for trader in stabilizers:
            vol = avg_volatility[trader]
            print(f"  {trader:15s} : {vol:5.2f}%")

        print(f"\nDESTABILIZERS (volatility > {threshold_pct}%):")
        for trader in destabilizers:
            vol = avg_volatility[trader]
            print(f"  {trader:15s} : {vol:5.2f}%")
        print()

        return (stabilizers, destabilizers)

    def analyze_volatility_efficiency_correlation(self) -> Dict[str, float]:
        """Analyze correlation between volatility and efficiency."""
        if self.all_results.empty:
            return {}

        # Overall correlation
        overall_corr = self.all_results[['price_volatility_pct', 'efficiency']].corr().iloc[0, 1]

        # Per-trader correlations
        trader_corrs = {}
        for trader in self.traders:
            trader_data = self.all_results[self.all_results['agent_type'] == trader]
            if len(trader_data) > 2:  # Need at least 3 points for correlation
                corr = trader_data[['price_volatility_pct', 'efficiency']].corr().iloc[0, 1]
                trader_corrs[trader] = corr

        print("\n" + "=" * 80)
        print("VOLATILITY-EFFICIENCY CORRELATION")
        print("=" * 80)
        print(f"\nOverall correlation: {overall_corr:.3f}")
        print("\nPer-trader correlations:")
        for trader, corr in sorted(trader_corrs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {trader:15s} : {corr:6.3f}")
        print()

        return {'overall': overall_corr, **trader_corrs}

    def generate_latex_table(self, output_file: Optional[str] = None) -> str:
        """
        Generate LaTeX table of volatility statistics.

        Args:
            output_file: Optional file path to save table

        Returns:
            LaTeX table string
        """
        if self.all_results.empty:
            return ""

        summary = self.analyze_by_trader()

        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Price Volatility by Trader Type}\n"
        latex += "\\label{tab:price_volatility}\n"
        latex += "\\begin{tabular}{lrrrr}\n"
        latex += "\\toprule\n"
        latex += "Trader & Avg Volatility (\\%) & Std Dev & Avg Price & Efficiency (\\%) \\\\\n"
        latex += "\\midrule\n"

        for trader, row in summary.iterrows():
            latex += f"{trader} & {row['avg_volatility_pct']:.2f} & {row['avg_std_dev']:.2f} & "
            latex += f"{row['avg_price']:.2f} & {row['avg_efficiency']:.2f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\multicolumn{5}{l}{\\footnotesize Volatility = (std\\_dev / mean) $\\times$ 100\\%} \\\\\n"
        latex += "\\multicolumn{5}{l}{\\footnotesize Lower volatility indicates more stable price discovery.} \\\\\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(latex)
            print(f"\nLaTeX table saved to: {output_file}")

        return latex

    def run_full_analysis(self, latex_output: Optional[str] = None):
        """Run complete volatility analysis pipeline."""
        print("\n" + "=" * 100)
        print("PRICE VOLATILITY ANALYSIS")
        print("=" * 100)

        # Load data
        self.load_results()

        # Run analyses
        self.analyze_by_trader()
        self.analyze_pure_vs_mixed()
        self.identify_stabilizers(threshold_pct=10.0)
        self.analyze_volatility_efficiency_correlation()

        # Generate LaTeX table if requested
        if latex_output:
            self.generate_latex_table(latex_output)

        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze price volatility patterns in Santa Fe tournament results'
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Tournament results directory (e.g., results/tournament_volatility_20251124_083936)'
    )
    parser.add_argument(
        '--latex',
        type=str,
        default=None,
        help='Optional output file for LaTeX table'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Volatility threshold for stabilizer classification (default: 10%%)'
    )

    args = parser.parse_args()

    analyzer = VolatilityAnalyzer(args.results_dir)
    analyzer.run_full_analysis(latex_output=args.latex)


if __name__ == '__main__':
    main()

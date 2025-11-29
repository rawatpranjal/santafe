#!/usr/bin/env python3
"""
Comprehensive Tournament Results Analyzer.

Aggregates results from all tournament experiments and generates summary reports.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class TournamentAnalyzer:
    """Analyzes and aggregates tournament results."""

    def __init__(self, results_dirs: List[str]):
        """
        Initialize analyzer with results directories.

        Args:
            results_dirs: List of tournament result directories
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.all_results = []
        self.traders = set()

    def load_all_results(self) -> pd.DataFrame:
        """Load all CSV results from tournament directories."""
        dfs = []

        for results_dir in self.results_dirs:
            if not results_dir.exists():
                print(f"Warning: {results_dir} does not exist")
                continue

            # Find all CSV files
            csv_files = list(results_dir.glob("**/*.csv"))
            print(f"Found {len(csv_files)} CSV files in {results_dir}")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['source_file'] = csv_file.name
                    df['category'] = results_dir.name.split('_')[1]  # Extract category
                    dfs.append(df)

                    # Track unique traders
                    if 'agent_type' in df.columns:
                        self.traders.update(df['agent_type'].unique())

                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

        if dfs:
            self.all_results = pd.concat(dfs, ignore_index=True)
            print(f"\nLoaded {len(self.all_results)} total experiments")
            print(f"Unique traders: {sorted(self.traders)}")
            return self.all_results
        else:
            print("No results loaded!")
            return pd.DataFrame()

    def analyze_pure_self_play(self) -> pd.DataFrame:
        """Analyze pure self-play efficiency."""
        if self.all_results.empty:
            return pd.DataFrame()

        pure_results = self.all_results[self.all_results['category'] == 'pure']

        if pure_results.empty:
            return pd.DataFrame()

        # Group by trader type
        summary = pure_results.groupby('agent_type').agg({
            'efficiency': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.sort_values('efficiency_mean', ascending=False)

        print("\n" + "="*80)
        print("PURE SELF-PLAY EFFICIENCY")
        print("="*80)
        print(summary)

        return summary

    def analyze_pairwise_comparisons(self) -> pd.DataFrame:
        """Analyze pairwise head-to-head results."""
        if self.all_results.empty:
            return pd.DataFrame()

        pairwise_results = self.all_results[self.all_results['category'] == 'pairwise']

        if pairwise_results.empty:
            return pd.DataFrame()

        print("\n" + "="*80)
        print("PAIRWISE COMPARISONS")
        print("="*80)
        print(f"Total pairwise experiments: {len(pairwise_results)}")

        # Extract trader pairs from source files (e.g., "gd_vs_zip.csv")
        pairwise_results['matchup'] = pairwise_results['source_file'].str.replace('.csv', '')

        summary = pairwise_results.groupby('matchup').agg({
            'efficiency': 'mean'
        }).round(2)

        summary = summary.sort_values('efficiency', ascending=False)
        print(summary.head(20))

        return summary

    def analyze_invasibility(self) -> pd.DataFrame:
        """Analyze 1v7 invasibility results."""
        if self.all_results.empty:
            return pd.DataFrame()

        invasibility_results = self.all_results[
            (self.all_results['category'] == 'seven') |
            (self.all_results['category'] == 'v')
        ]

        if invasibility_results.empty:
            return pd.DataFrame()

        print("\n" + "="*80)
        print("INVASIBILITY (1v7) RESULTS")
        print("="*80)
        print(f"Total invasibility experiments: {len(invasibility_results)}")

        return invasibility_results

    def analyze_mixed_strategies(self) -> pd.DataFrame:
        """Analyze mixed strategy tournaments."""
        if self.all_results.empty:
            return pd.DataFrame()

        mixed_results = self.all_results[self.all_results['category'] == 'mixed']

        if mixed_results.empty:
            return pd.DataFrame()

        print("\n" + "="*80)
        print("MIXED STRATEGY RESULTS")
        print("="*80)

        summary = mixed_results.groupby('source_file').agg({
            'efficiency': 'mean'
        }).round(2)

        print(summary)

        return summary

    def create_efficiency_ranking(self) -> pd.DataFrame:
        """Create overall trader efficiency ranking."""
        if self.all_results.empty:
            return pd.DataFrame()

        # Focus on pure self-play for baseline ranking
        pure_results = self.all_results[self.all_results['category'] == 'pure']

        if pure_results.empty:
            return pd.DataFrame()

        ranking = pure_results.groupby('agent_type').agg({
            'efficiency': ['mean', 'std', 'count']
        }).round(2)

        ranking.columns = ['efficiency_mean', 'efficiency_std', 'n_experiments']
        ranking = ranking.sort_values('efficiency_mean', ascending=False)
        ranking['rank'] = range(1, len(ranking) + 1)

        print("\n" + "="*80)
        print("TRADER EFFICIENCY RANKING (Pure Self-Play)")
        print("="*80)
        print(ranking)

        return ranking

    def validate_lin_perry(self) -> Dict:
        """Validate Lin and Perry specific metrics."""
        if self.all_results.empty:
            return {}

        validation = {}

        # Lin self-play
        lin_pure = self.all_results[
            (self.all_results['category'] == 'pure') &
            (self.all_results['agent_type'] == 'Lin')
        ]

        if not lin_pure.empty:
            lin_eff = lin_pure['efficiency'].mean()
            lin_std = lin_pure['efficiency'].std()
            validation['lin_self_play'] = {
                'efficiency': lin_eff,
                'std': lin_std,
                'target': 99.85,
                'validated': abs(lin_eff - 99.85) < 5.0
            }

        # Perry self-play
        perry_pure = self.all_results[
            (self.all_results['category'] == 'pure') &
            (self.all_results['agent_type'] == 'Perry')
        ]

        if not perry_pure.empty:
            perry_eff = perry_pure['efficiency'].mean()
            perry_std = perry_pure['efficiency'].std()
            validation['perry_self_play'] = {
                'efficiency': perry_eff,
                'std': perry_std,
                'target': 82.00,
                'validated': abs(perry_eff - 82.00) < 10.0
            }

        print("\n" + "="*80)
        print("LIN & PERRY VALIDATION")
        print("="*80)
        for trader, metrics in validation.items():
            print(f"\n{trader.upper()}:")
            print(f"  Efficiency: {metrics['efficiency']:.2f}% ± {metrics['std']:.2f}%")
            print(f"  Target: {metrics['target']:.2f}%")
            print(f"  Validated: {'✅' if metrics['validated'] else '❌'}")

        return validation

    def generate_summary_report(self, output_file: str = "tournament_summary.md"):
        """Generate comprehensive markdown summary report."""
        if self.all_results.empty:
            print("No results to report!")
            return

        with open(output_file, 'w') as f:
            f.write("# Tournament Results Summary\n\n")
            f.write(f"**Total Experiments:** {len(self.all_results)}\n")
            f.write(f"**Unique Traders:** {len(self.traders)}\n")
            f.write(f"**Categories:** {sorted(self.all_results['category'].unique())}\n\n")

            # Pure self-play
            f.write("## Pure Self-Play Efficiency\n\n")
            pure_summary = self.analyze_pure_self_play()
            if not pure_summary.empty:
                f.write(pure_summary.to_markdown())
                f.write("\n\n")

            # Lin & Perry validation
            f.write("## Lin & Perry Validation\n\n")
            validation = self.validate_lin_perry()
            for trader, metrics in validation.items():
                f.write(f"**{trader.upper()}:**\n")
                f.write(f"- Efficiency: {metrics['efficiency']:.2f}% ± {metrics['std']:.2f}%\n")
                f.write(f"- Target: {metrics['target']:.2f}%\n")
                f.write(f"- Validated: {'✅' if metrics['validated'] else '❌'}\n\n")

            # Efficiency ranking
            f.write("## Trader Efficiency Ranking\n\n")
            ranking = self.create_efficiency_ranking()
            if not ranking.empty:
                f.write(ranking.to_markdown())
                f.write("\n\n")

        print(f"\nSummary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results")
    parser.add_argument(
        '--results-dirs',
        nargs='+',
        help='Directories containing tournament results'
    )
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Automatically discover result directories in results/'
    )
    parser.add_argument(
        '--output',
        default='tournament_summary.md',
        help='Output summary file'
    )

    args = parser.parse_args()

    # Auto-discover results directories
    if args.auto_discover:
        results_base = Path('results')
        if results_base.exists():
            results_dirs = [
                str(d) for d in results_base.iterdir()
                if d.is_dir() and d.name.startswith('tournament_')
            ]
            print(f"Auto-discovered {len(results_dirs)} result directories")
        else:
            print("No results/ directory found")
            results_dirs = []
    else:
        results_dirs = args.results_dirs or []

    if not results_dirs:
        print("No results directories specified!")
        print("Use --auto-discover or provide --results-dirs")
        return

    # Run analysis
    analyzer = TournamentAnalyzer(results_dirs)
    analyzer.load_all_results()

    if not analyzer.all_results.empty:
        analyzer.analyze_pure_self_play()
        analyzer.analyze_pairwise_comparisons()
        analyzer.analyze_mixed_strategies()
        analyzer.analyze_invasibility()
        analyzer.create_efficiency_ranking()
        analyzer.validate_lin_perry()
        analyzer.generate_summary_report(args.output)
    else:
        print("No results to analyze!")


if __name__ == '__main__':
    main()

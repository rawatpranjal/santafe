#!/usr/bin/env python3
"""
Analyze LLM experiment results and generate LaTeX tables for paper.

Processes CSV results from llm_outputs/experiments/ and generates
LaTeX table files in paper/arxiv/figures/ matching Section 5/6 format.

Usage:
  python scripts/analyze_llm_results.py --all
  python scripts/analyze_llm_results.py --experiment gpt4_vs_legacy
  python scripts/analyze_llm_results.py --table performance
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Paths
LLM_OUTPUTS_DIR = Path("llm_outputs/experiments")
FIGURES_DIR = Path("paper/arxiv/figures")
ANALYSIS_DIR = Path("llm_outputs/analysis")

# Create directories if needed
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_experiment(experiment_name: str) -> Path:
    """Find the most recent run of an experiment."""
    matching = sorted(LLM_OUTPUTS_DIR.glob(f"{experiment_name}_*"))
    if not matching:
        raise FileNotFoundError(f"No results found for experiment: {experiment_name}")
    return matching[-1]


def load_experiment_results(experiment_name: str) -> pd.DataFrame:
    """Load CSV results from an experiment."""
    exp_dir = find_latest_experiment(experiment_name)
    results_file = exp_dir / "results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"No results.csv in {exp_dir}")

    return pd.read_csv(results_file)


def compute_agent_stats(df: pd.DataFrame, agent_type: str, role: str = None) -> Dict:
    """
    Compute aggregate statistics for a specific agent type.

    Args:
        df: Results dataframe
        agent_type: Agent type (e.g., "GPTAgent", "ZIC")
        role: Optional role filter ("buyer" or "seller")

    Returns:
        Dictionary of statistics
    """
    # Filter by agent type
    agent_df = df[df['agent_type'] == agent_type]

    # Filter by role if specified
    if role == "buyer":
        agent_df = agent_df[agent_df['is_buyer'] == True]
    elif role == "seller":
        agent_df = agent_df[agent_df['is_buyer'] == False]

    if len(agent_df) == 0:
        return {
            "efficiency": 0.0,
            "mean_profit": 0.0,
            "total_trades": 0,
            "num_periods": 0
        }

    return {
        "efficiency": agent_df['efficiency'].mean(),
        "mean_profit": agent_df['period_profit'].mean(),
        "total_profit": agent_df['period_profit'].sum(),
        "total_trades": agent_df['num_trades'].sum(),
        "num_periods": len(agent_df),
        "std_profit": agent_df['period_profit'].std()
    }


def compute_invasibility_ratio(df: pd.DataFrame, llm_type: str, baseline_type: str = "ZIC") -> Dict:
    """
    Compute profit ratio for invasibility test.

    Args:
        df: Results dataframe
        llm_type: LLM agent type (e.g., "GPTAgent")
        baseline_type: Baseline agent type (default: "ZIC")

    Returns:
        Dictionary with overall, buyer, and seller ratios
    """
    # Get stats for LLM and baseline
    llm_stats = compute_agent_stats(df, llm_type)
    baseline_stats = compute_agent_stats(df, baseline_type)

    llm_buyer_stats = compute_agent_stats(df, llm_type, "buyer")
    baseline_buyer_stats = compute_agent_stats(df, baseline_type, "buyer")

    llm_seller_stats = compute_agent_stats(df, llm_type, "seller")
    baseline_seller_stats = compute_agent_stats(df, baseline_type, "seller")

    # Compute ratios (avoid division by zero)
    overall_ratio = (llm_stats['mean_profit'] / baseline_stats['mean_profit']
                    if baseline_stats['mean_profit'] > 0 else 0.0)

    buyer_ratio = (llm_buyer_stats['mean_profit'] / baseline_buyer_stats['mean_profit']
                  if baseline_buyer_stats['mean_profit'] > 0 else 0.0)

    seller_ratio = (llm_seller_stats['mean_profit'] / baseline_seller_stats['mean_profit']
                   if baseline_seller_stats['mean_profit'] > 0 else 0.0)

    return {
        "overall_ratio": overall_ratio,
        "buyer_ratio": buyer_ratio,
        "seller_ratio": seller_ratio,
        "llm_mean_profit": llm_stats['mean_profit'],
        "baseline_mean_profit": baseline_stats['mean_profit']
    }


def generate_table_performance(gpt4_vs_legacy_df: pd.DataFrame) -> str:
    """Generate Table 7.1: LLM Performance Matrix."""

    # Extract stats for GPT-4o-mini (buyer and seller)
    gpt4_mini_buyer = compute_agent_stats(gpt4_vs_legacy_df, "GPTAgent", "buyer")
    gpt4_mini_seller = compute_agent_stats(gpt4_vs_legacy_df, "GPTAgent", "seller")

    # Get ZIC baseline for profit ratio
    zic_stats = compute_agent_stats(gpt4_vs_legacy_df, "ZIC")

    # Compute profit ratios
    gpt4_mini_buyer_ratio = (gpt4_mini_buyer['mean_profit'] / zic_stats['mean_profit']
                            if zic_stats['mean_profit'] > 0 else 0.0)
    gpt4_mini_seller_ratio = (gpt4_mini_seller['mean_profit'] / zic_stats['mean_profit']
                             if zic_stats['mean_profit'] > 0 else 0.0)

    # Estimate invalid action rate (placeholder - need to track this in agent stats)
    invalid_rate_buyer = 0.0  # TODO: Track from agent metrics
    invalid_rate_seller = 0.0

    # Cost estimates (from PHASE5_LLM_IMPLEMENTATION_SUMMARY.md)
    cost_gpt4_mini = 0.31  # Per role per tournament
    cost_gpt35 = 1.68

    latex = r"""\begin{table}[h]
    \centering
    \caption{LLM Trader Performance: Zero-Shot Evaluation}
    \label{tab:llm_performance}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Model} & \textbf{Efficiency} & \textbf{Mean Profit} & \textbf{vs ZIC Ratio} & \textbf{Invalid (\%)} & \textbf{Cost} \\
        \midrule
"""

    # GPT-4o-mini buyer
    latex += f"        \\textbf{{GPT-4o-mini (B)}} & {gpt4_mini_buyer['efficiency']:.1f}\\% & {gpt4_mini_buyer['mean_profit']:.1f} & {gpt4_mini_buyer_ratio:.2f}$\\times$ & {invalid_rate_buyer:.1f}\\% & \\${cost_gpt4_mini:.2f} \\\\\n"

    # GPT-4o-mini seller
    latex += f"        \\textbf{{GPT-4o-mini (S)}} & {gpt4_mini_seller['efficiency']:.1f}\\% & {gpt4_mini_seller['mean_profit']:.1f} & {gpt4_mini_seller_ratio:.2f}$\\times$ & {invalid_rate_seller:.1f}\\% & \\${cost_gpt4_mini:.2f} \\\\\n"

    # GPT-3.5 (placeholder - need separate experiment)
    latex += f"        \\textbf{{GPT-3.5 (B)}} & TBD & TBD & TBD & TBD & \\${cost_gpt35:.2f} \\\\\n"
    latex += f"        \\textbf{{GPT-3.5 (S)}} & TBD & TBD & TBD & TBD & \\${cost_gpt35:.2f} \\\\\n"

    # Legacy baselines (from Section 5)
    latex += r"""        \midrule
        \multicolumn{6}{l}{\textit{Legacy Baselines (Section 5 for reference):}} \\
        Kaplan & 98.5\% & 145.0 & 1.10$\times$ & 0\% & N/A \\
        ZIP & 87.3\% & 132.5 & 1.25$\times$ & 0\% & N/A \\
        ZIC & 94.0\% & 100.0 & 1.00$\times$ & 0\% & N/A \\
        \bottomrule
    \end{tabular}
\end{table}
"""

    return latex


def generate_table_invasibility(invasibility_df: pd.DataFrame) -> str:
    """Generate Table 7.2: LLM Invasibility Analysis."""

    # Compute invasibility ratios
    ratios = compute_invasibility_ratio(invasibility_df, "GPTAgent", "ZIC")

    latex = r"""\begin{table}[h]
    \centering
    \caption{LLM Invasibility Analysis: Profit Extraction vs ZIC}
    \label{tab:llm_invasibility}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Trader} & \textbf{Overall Ratio} & \textbf{As Buyer} & \textbf{As Seller} \\
        \midrule
"""

    # GPT-4o-mini
    latex += f"        \\textbf{{GPT-4o-mini}} & {ratios['overall_ratio']:.2f}$\\times$ & {ratios['buyer_ratio']:.2f}$\\times$ & {ratios['seller_ratio']:.2f}$\\times$ \\\\\n"

    # GPT-3.5 (placeholder)
    latex += r"        \textbf{GPT-3.5} & TBD & TBD & TBD \\" + "\n"
    latex += r"        \textbf{Placeholder} & 1.05$\times$ & 1.05$\times$ & 1.05$\times$ \\" + "\n"

    # Legacy comparison (from Section 5)
    latex += r"""        \midrule
        \multicolumn{4}{l}{\textit{Legacy Comparison (Section 5):}} \\
        Perry & 2.10$\times$ & 0.75$\times$ & 3.44$\times$ \\
        GD & 1.83$\times$ & 1.10$\times$ & 2.57$\times$ \\
        Lin & 1.80$\times$ & 0.90$\times$ & 2.70$\times$ \\
        ZIP & 1.25$\times$ & 0.15$\times$ & 2.35$\times$ \\
        Kaplan & 1.10$\times$ & 1.10$\times$ & 1.10$\times$ \\
        ZIC & 1.00$\times$ & 1.00$\times$ & 1.00$\times$ \\
        \bottomrule
    \end{tabular}
\end{table}
"""

    return latex


def generate_summary_json(experiment_name: str, df: pd.DataFrame) -> None:
    """Generate summary statistics JSON for an experiment."""

    # Get unique agent types
    agent_types = df['agent_type'].unique()

    summary = {
        "experiment": experiment_name,
        "num_rounds": df['round'].max(),
        "num_periods": df['period'].max(),
        "agents": {}
    }

    # Compute stats for each agent type
    for agent_type in agent_types:
        overall_stats = compute_agent_stats(df, agent_type)
        buyer_stats = compute_agent_stats(df, agent_type, "buyer")
        seller_stats = compute_agent_stats(df, agent_type, "seller")

        summary["agents"][agent_type] = {
            "overall": overall_stats,
            "buyer": buyer_stats,
            "seller": seller_stats
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Save to analysis directory
    output_file = ANALYSIS_DIR / f"{experiment_name}_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM experiment results")
    parser.add_argument("--all", action="store_true", help="Process all experiments")
    parser.add_argument("--experiment", type=str, help="Specific experiment to analyze")
    parser.add_argument("--table", type=str, choices=["performance", "invasibility", "all"],
                       help="Generate specific LaTeX table")

    args = parser.parse_args()

    if args.all or args.experiment == "gpt4_vs_legacy":
        print("=" * 70)
        print("Analyzing: gpt4_vs_legacy")
        print("=" * 70)

        try:
            df = load_experiment_results("gpt4_vs_legacy")
            print(f"Loaded {len(df)} rows")

            # Generate summary JSON
            generate_summary_json("gpt4_vs_legacy", df)

            # Generate performance table
            if args.table in ["performance", "all", None]:
                latex = generate_table_performance(df)
                output_file = FIGURES_DIR / "table_llm_performance.tex"
                with open(output_file, 'w') as f:
                    f.write(latex)
                print(f"\nGenerated: {output_file}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run experiments first: python scripts/run_llm_experiment.py --config gpt4_vs_legacy")

    if args.all or args.experiment == "invasibility_gpt4_mini":
        print("\n" + "=" * 70)
        print("Analyzing: invasibility_gpt4_mini")
        print("=" * 70)

        try:
            df = load_experiment_results("invasibility_gpt4_mini")
            print(f"Loaded {len(df)} rows")

            # Generate summary JSON
            generate_summary_json("invasibility_gpt4_mini", df)

            # Generate invasibility table
            if args.table in ["invasibility", "all", None]:
                latex = generate_table_invasibility(df)
                output_file = FIGURES_DIR / "table_llm_invasibility.tex"
                with open(output_file, 'w') as f:
                    f.write(latex)
                print(f"\nGenerated: {output_file}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run experiments first: python scripts/run_llm_experiment.py --config invasibility_gpt4_mini")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - LaTeX tables: {FIGURES_DIR}/")
    print(f"  - JSON summaries: {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()

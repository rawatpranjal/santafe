"""Generate LaTeX tables for PPO Section 7 (matching Part 2 format)."""

import glob
import json
from pathlib import Path


def find_latest_results(results_dir: Path, prefix: str) -> Path:
    """Find the most recent results file with given prefix."""
    pattern = str(results_dir / f"{prefix}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.json files found in {results_dir}")
    return Path(files[-1])


def generate_control_efficiency_table(control_data: dict, output_path: Path):
    """Generate PPO control efficiency table matching table_control.tex format."""
    envs = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]

    latex = r"""\begin{table}[H]
\centering
\caption{PPO Against Control: 1 PPO vs 7 ZIC (Efficiency \%)}
\label{tab:ppo_control}
\begin{tabular}{lrrrrrrrrrr}
\toprule
Strategy & BASE & BBBS & BSSS & EQL & RAN & PER & SHRT & TOK & SML & LAD \\
\midrule
"""

    # Build PPO row
    row = "PPO"
    for env in envs:
        if env in control_data:
            mean = control_data[env]["efficiency_mean"]
            std = control_data[env]["efficiency_std"]
            row += f" & {mean:.0f}$\\pm${std:.0f}"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    latex += r"""\bottomrule
\multicolumn{11}{l}{\footnotesize 5 seeds, 50 rounds each. Mean $\pm$ std efficiency.} \\
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Generated: {output_path}")


def generate_control_volatility_table(control_data: dict, output_path: Path):
    """Generate PPO control volatility table."""
    envs = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]

    latex = r"""\begin{table}[H]
\centering
\caption{PPO Against Control: Price Volatility (\%)}
\label{tab:ppo_control_volatility}
\begin{tabular}{lrrrrrrrrrr}
\toprule
Strategy & BASE & BBBS & BSSS & EQL & RAN & PER & SHRT & TOK & SML & LAD \\
\midrule
"""

    row = "PPO"
    for env in envs:
        if env in control_data:
            mean = control_data[env]["volatility_mean"]
            row += f" & {mean:.1f}"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    latex += r"""\bottomrule
\multicolumn{11}{l}{\footnotesize 5 seeds, 50 rounds each. Price volatility = std(prices)/mean(prices).} \\
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Generated: {output_path}")


def generate_invasibility_table(control_data: dict, output_path: Path):
    """Generate PPO invasibility table matching table_invasibility.tex format."""
    # Exclude RAN due to extreme values from near-zero ZIC profits
    envs = ["BASE", "BBBS", "BSSS", "EQL", "PER", "SHRT", "TOK", "SML", "LAD"]

    latex = r"""\begin{table}[H]
\centering
\caption{PPO Control Profit Ratios (Invasibility): PPO Profit / ZIC Profit}
\label{tab:ppo_invasibility}
\begin{tabular}{lrrrrrrrrr}
\toprule
Strategy & BASE & BBBS & BSSS & EQL & PER & SHRT & TOK & SML & LAD \\
\midrule
"""

    row = "PPO"
    for env in envs:
        if env in control_data:
            inv = control_data[env]["invasibility"]
            # Cap extreme values for display
            if inv > 100:
                row += f" & {inv:.0f}x"
            else:
                row += f" & {inv:.2f}x"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    latex += r"""\bottomrule
\multicolumn{10}{l}{\footnotesize Ratio $>$1.0 = PPO exploits ZIC. RAN excluded (extreme profit ratios).} \\
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Generated: {output_path}")


def generate_pairwise_table(pairwise_data: dict, output_path: Path):
    """Generate PPO pairwise competition table."""
    opponents = ["ZIC", "ZIP", "Skeleton", "Kaplan"]

    latex = r"""\begin{table}[H]
\centering
\caption{PPO Pairwise Competition: Mixed Market Performance (2 PPO + 2 Opponent per side)}
\label{tab:ppo_pairwise}
\begin{tabular}{lrrrr}
\toprule
\textbf{Metric} & \textbf{PPO vs ZIC} & \textbf{PPO vs ZIP} & \textbf{PPO vs Skeleton} & \textbf{PPO vs Kaplan} \\
\midrule
"""

    # Efficiency row
    row = "Efficiency (mean$\\pm$std)"
    for opp in opponents:
        if opp in pairwise_data:
            mean = pairwise_data[opp]["efficiency_mean"]
            std = pairwise_data[opp]["efficiency_std"]
            row += f" & {mean:.1f}$\\pm${std:.1f}\\%"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    # PPO profit row
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{5}{l}{\textit{Mean Profit per Agent}}" + r" \\" + "\n"

    row = "PPO Profit"
    for opp in opponents:
        if opp in pairwise_data:
            mean = pairwise_data[opp]["ppo_profit_mean"]
            row += f" & {mean:.0f}"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    # Opponent profit row
    row = "Opponent Profit"
    for opp in opponents:
        if opp in pairwise_data:
            mean = pairwise_data[opp]["opponent_profit_mean"]
            row += f" & {mean:.0f}"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    # Profit ratio row
    row = "PPO/Opponent Ratio"
    for opp in opponents:
        if opp in pairwise_data:
            ratio = pairwise_data[opp]["profit_ratio"]
            row += f" & {ratio:.2f}x"
        else:
            row += " & --"
    row += r" \\"
    latex += row + "\n"

    latex += r"""\bottomrule
\multicolumn{5}{l}{\footnotesize 5 seeds, 50 rounds each in BASE environment. Ratio $>$1.0 = PPO outperforms opponent.} \\
\end{tabular}
\end{table}
"""

    output_path.write_text(latex)
    print(f"Generated: {output_path}")


def main():
    results_dir = Path("/Users/pranjal/Code/santafe-1/results/ppo_section7")
    figures_dir = Path("/Users/pranjal/Code/santafe-1/paper/arxiv/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load control results
    try:
        control_path = find_latest_results(results_dir, "control_results")
        print(f"Loading control results from: {control_path}")
        with open(control_path) as f:
            control_data = json.load(f)

        generate_control_efficiency_table(control_data, figures_dir / "table_ppo_control.tex")
        generate_control_volatility_table(
            control_data, figures_dir / "table_ppo_control_volatility.tex"
        )
        generate_invasibility_table(control_data, figures_dir / "table_ppo_invasibility.tex")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    # Load pairwise results
    try:
        pairwise_path = find_latest_results(results_dir, "pairwise_results")
        print(f"Loading pairwise results from: {pairwise_path}")
        with open(pairwise_path) as f:
            pairwise_data = json.load(f)

        generate_pairwise_table(pairwise_data, figures_dir / "table_ppo_pairwise.tex")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    print("\nDone! Tables saved to paper/arxiv/figures/")


if __name__ == "__main__":
    main()

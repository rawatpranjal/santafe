#!/usr/bin/env python3
"""
Calculate complete invasibility results for all 10 Santa Fe traders.
Generates LaTeX table for paper.
"""

import pandas as pd
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results/tournament_all_v_all_20251124_042520/one_v_seven")
OUTPUT_FILE = Path("paper/arxiv/figures/table_invasibility.tex")

# All 10 traders to analyze
TRADERS = [
    "zi", "zic", "zi2", "zip", "gd",
    "kaplan", "lin", "jacobson", "perry", "skeleton"
]

# Display names for table
TRADER_NAMES = {
    "zi": "ZI",
    "zic": "ZIC",
    "zi2": "ZI2",
    "zip": "ZIP",
    "gd": "GD",
    "kaplan": "Kaplan",
    "lin": "Lin",
    "jacobson": "Jacobson",
    "perry": "Perry",
    "skeleton": "Skeleton"
}


def calculate_profit_ratio(trader):
    """Calculate profit ratio for one trader vs 7 ZIC opponents."""

    # Load results CSV
    csv_path = RESULTS_DIR / f"{trader}_1v7_zic" / "results.csv"

    if not csv_path.exists():
        print(f"WARNING: No data for {trader} at {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Market is 8v8:
    # - Agent IDs 1-8: Buyers (agent 1 = tested trader, 2-8 = ZIC)
    # - Agent IDs 9-16: Sellers (agent 9 = tested trader, 10-16 = ZIC)

    # Tested trader as BUYER (agent_id = 1)
    trader_buyer_profit = df[df['agent_id'] == 1]['period_profit'].sum()

    # Tested trader as SELLER (agent_id = 9)
    trader_seller_profit = df[df['agent_id'] == 9]['period_profit'].sum()

    # ZIC opponents as BUYERS (agent_ids 2-8)
    zic_buyer_profits = []
    for agent_id in [2, 3, 4, 5, 6, 7, 8]:
        profit = df[df['agent_id'] == agent_id]['period_profit'].sum()
        zic_buyer_profits.append(profit)
    mean_zic_buyer = sum(zic_buyer_profits) / 7

    # ZIC opponents as SELLERS (agent_ids 10-16)
    zic_seller_profits = []
    for agent_id in [10, 11, 12, 13, 14, 15, 16]:
        profit = df[df['agent_id'] == agent_id]['period_profit'].sum()
        zic_seller_profits.append(profit)
    mean_zic_seller = sum(zic_seller_profits) / 7

    # Calculate ratios (handle division by zero/negative)
    if mean_zic_buyer == 0:
        buyer_ratio = float('inf') if trader_buyer_profit > 0 else 0.0
    else:
        buyer_ratio = trader_buyer_profit / mean_zic_buyer

    if mean_zic_seller == 0:
        seller_ratio = float('inf') if trader_seller_profit > 0 else 0.0
    else:
        seller_ratio = trader_seller_profit / mean_zic_seller

    # Overall is average of the two ratios
    overall_ratio = (buyer_ratio + seller_ratio) / 2

    return {
        'trader': trader,
        'name': TRADER_NAMES[trader],
        'overall': overall_ratio,
        'buyer': buyer_ratio,
        'seller': seller_ratio,
        'trader_buyer_profit': trader_buyer_profit,
        'trader_seller_profit': trader_seller_profit,
        'zic_buyer_profit': mean_zic_buyer,
        'zic_seller_profit': mean_zic_seller
    }


def generate_latex_table(results):
    """Generate LaTeX table code."""

    # Sort by overall ratio descending
    results_sorted = sorted(results, key=lambda x: x['overall'], reverse=True)

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Invasibility Test: 1 Trader vs 7 ZIC Agents (8v8 Market)}")
    latex.append(r"\label{tab:invasibility}")
    latex.append(r"\begin{tabular}{lrrr}")
    latex.append(r"\toprule")
    latex.append(r"Trader & Overall Ratio & As Buyer & As Seller \\")
    latex.append(r"\midrule")

    for r in results_sorted:
        # Format ratios
        overall_str = f"{r['overall']:.2f}x"
        buyer_str = f"{r['buyer']:.2f}x"
        seller_str = f"{r['seller']:.2f}x"

        # Handle infinity/extreme values
        if r['overall'] > 999:
            overall_str = ">999x"
        if r['buyer'] > 999:
            buyer_str = ">999x"
        if r['seller'] > 999:
            seller_str = ">999x"

        latex.append(f"{r['name']} & {overall_str} & {buyer_str} & {seller_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\multicolumn{4}{l}{\footnotesize Ratio = (Trader profit per agent) / (ZIC profit per agent)} \\")
    latex.append(r"\multicolumn{4}{l}{\footnotesize Values >1.0 indicate profit extraction from ZIC population.} \\")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    return "\n".join(latex)


def main():
    print("=" * 80)
    print("CALCULATING INVASIBILITY RATIOS FOR ALL 10 TRADERS")
    print("=" * 80)
    print()

    results = []

    for trader in TRADERS:
        print(f"Processing {TRADER_NAMES[trader]}...", end=" ")

        ratio_data = calculate_profit_ratio(trader)

        if ratio_data is None:
            print("SKIP (no data)")
            continue

        results.append(ratio_data)
        print(f"Overall: {ratio_data['overall']:.2f}x  "
              f"(Buyer: {ratio_data['buyer']:.2f}x, Seller: {ratio_data['seller']:.2f}x)")

    print()
    print(f"Processed {len(results)}/10 traders")
    print()

    if len(results) == 0:
        print("ERROR: No results calculated!")
        return

    # Generate LaTeX table
    latex_code = generate_latex_table(results)

    # Write to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(latex_code)

    print(f"Generated LaTeX table: {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("TABLE PREVIEW")
    print("=" * 80)
    print(latex_code)
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Sort by overall for summary
    results_sorted = sorted(results, key=lambda x: x['overall'], reverse=True)

    print(f"Top Performer: {results_sorted[0]['name']} ({results_sorted[0]['overall']:.2f}x)")
    print(f"Worst Performer: {results_sorted[-1]['name']} ({results_sorted[-1]['overall']:.2f}x)")
    print()
    print("Buyer vs Seller Asymmetry (largest gaps):")
    asymmetries = [(r['name'], abs(r['buyer'] - r['seller'])) for r in results]
    asymmetries_sorted = sorted(asymmetries, key=lambda x: x[1], reverse=True)
    for name, gap in asymmetries_sorted[:3]:
        print(f"  {name}: {gap:.2f}x gap")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate round-robin tournament results for Santa Fe traders."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Santa Fe 1991 roster (12 traders)
SANTA_FE_TRADERS = [
    "ZIC",
    "Skeleton",
    "Kaplan",
    "Ringuette",
    "Gamer",
    "Perry",
    "Ledyard",
    "BGAN",
    "Staecker",
    "Jacobson",
    "Lin",
    "Breton",
]

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]


def load_rr_results(results_dir: Path) -> dict:
    """Load all round-robin results."""
    results = {}
    for env in ENVS:
        exp_dir = results_dir / f"p2_rr_mixed_{env}"
        csv_path = exp_dir / "results.csv"
        if csv_path.exists():
            results[env] = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results[env].append(row)
    return results


def normalize_strategy_name(name: str) -> str:
    """Normalize strategy name (e.g., ZIC1 -> ZIC)."""
    # Remove trailing numbers from trader names like ZIC1, ZIC2
    if name.startswith("ZIC") and name != "ZIC":
        return "ZIC"
    return name


def analyze_profits_and_ranks(results: dict) -> tuple[dict, dict, dict]:
    """Analyze profits and compute ranks by environment."""
    # profit_by_env_strategy[env][strategy] = list of profits
    profit_by_env_strategy = {env: defaultdict(list) for env in ENVS}
    rank_by_env_strategy = {env: defaultdict(list) for env in ENVS}

    for env, rows in results.items():
        # Group by (round, agent_id) and sum profits across periods
        round_agent_profits = defaultdict(lambda: defaultdict(float))
        round_agent_strategy = {}

        for row in rows:
            round_num = int(row["round"])
            agent_id = int(row["agent_id"])
            strategy = normalize_strategy_name(row["agent_type"])
            profit = float(row["period_profit"])

            round_agent_profits[round_num][agent_id] += profit
            round_agent_strategy[(round_num, agent_id)] = strategy

        # For each round, compute ranks
        for round_num, agent_profits in round_agent_profits.items():
            # Build list of (strategy, total_profit)
            entries = []
            for agent_id, profit in agent_profits.items():
                strategy = round_agent_strategy[(round_num, agent_id)]
                entries.append((strategy, profit))

            # Sort by profit descending
            sorted_entries = sorted(entries, key=lambda x: -x[1])

            for rank, (strategy, profit) in enumerate(sorted_entries, 1):
                profit_by_env_strategy[env][strategy].append(profit)
                rank_by_env_strategy[env][strategy].append(rank)

    # Compute summary stats
    profit_summary = {}
    rank_summary = {}

    for env in ENVS:
        profit_summary[env] = {}
        rank_summary[env] = {}
        for strategy in SANTA_FE_TRADERS:
            profits = profit_by_env_strategy[env][strategy]
            ranks = rank_by_env_strategy[env][strategy]
            if profits:
                profit_summary[env][strategy] = {
                    "mean": np.mean(profits),
                    "std": np.std(profits),
                    "n": len(profits),
                }
                rank_summary[env][strategy] = {
                    "mean": np.mean(ranks),
                    "std": np.std(ranks),
                    "n": len(ranks),
                }

    # Overall stats
    overall_rank = {}
    overall_wins = defaultdict(int)

    for strategy in SANTA_FE_TRADERS:
        all_ranks = []
        for env in ENVS:
            if strategy in rank_summary[env]:
                all_ranks.extend(rank_by_env_strategy[env][strategy])
        if all_ranks:
            overall_rank[strategy] = np.mean(all_ranks)
            # Count rank=1 as wins
            overall_wins[strategy] = sum(1 for r in all_ranks if r == 1)

    return profit_summary, rank_summary, {"avg_rank": overall_rank, "wins": dict(overall_wins)}


def format_profit_table(profit_summary: dict) -> str:
    """Format profit table for markdown."""
    lines = []
    header = "| Env |" + " | ".join(SANTA_FE_TRADERS[:6]) + " |"
    lines.append(header)
    lines.append("|-----" + "|--------" * 6 + "|")

    for env in ENVS:
        row = f"| {env.upper()} |"
        for strat in SANTA_FE_TRADERS[:6]:
            if strat in profit_summary[env]:
                p = profit_summary[env][strat]
                row += f" {p['mean']:.0f}±{p['std']:.0f} |"
            else:
                row += " - |"
        lines.append(row)

    # Second table for remaining traders
    lines.append("")
    header2 = "| Env |" + " | ".join(SANTA_FE_TRADERS[6:]) + " |"
    lines.append(header2)
    lines.append("|-----" + "|--------" * 6 + "|")

    for env in ENVS:
        row = f"| {env.upper()} |"
        for strat in SANTA_FE_TRADERS[6:]:
            if strat in profit_summary[env]:
                p = profit_summary[env][strat]
                row += f" {p['mean']:.0f}±{p['std']:.0f} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def format_rank_table(rank_summary: dict) -> str:
    """Format rank table for markdown."""
    lines = []
    header = "| Env |" + " | ".join(SANTA_FE_TRADERS[:6]) + " |"
    lines.append(header)
    lines.append("|-----" + "|--------" * 6 + "|")

    for env in ENVS:
        row = f"| {env.upper()} |"
        for strat in SANTA_FE_TRADERS[:6]:
            if strat in rank_summary[env]:
                r = rank_summary[env][strat]
                row += f" {r['mean']:.1f}±{r['std']:.1f} |"
            else:
                row += " - |"
        lines.append(row)

    # Second table
    lines.append("")
    header2 = "| Env |" + " | ".join(SANTA_FE_TRADERS[6:]) + " |"
    lines.append(header2)
    lines.append("|-----" + "|--------" * 6 + "|")

    for env in ENVS:
        row = f"| {env.upper()} |"
        for strat in SANTA_FE_TRADERS[6:]:
            if strat in rank_summary[env]:
                r = rank_summary[env][strat]
                row += f" {r['mean']:.1f}±{r['std']:.1f} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def format_tournament_summary(rank_summary: dict, overall: dict) -> str:
    """Format tournament summary table."""
    lines = []
    lines.append("| Strategy | Avg Rank | Wins | Best Env | Worst Env |")
    lines.append("|----------|----------|------|----------|-----------|")

    # Sort by avg rank
    sorted_strats = sorted(overall["avg_rank"].items(), key=lambda x: x[1])

    for strat, avg_rank in sorted_strats:
        wins = overall["wins"].get(strat, 0)

        # Find best/worst env
        best_env, best_rank = None, 100
        worst_env, worst_rank = None, 0
        for env in ENVS:
            if strat in rank_summary[env]:
                r = rank_summary[env][strat]["mean"]
                if r < best_rank:
                    best_rank = r
                    best_env = env.upper()
                if r > worst_rank:
                    worst_rank = r
                    worst_env = env.upper()

        best_str = f"{best_env} ({best_rank:.1f})" if best_env else "-"
        worst_str = f"{worst_env} ({worst_rank:.1f})" if worst_env else "-"

        lines.append(f"| {strat} | {avg_rank:.2f} | {wins} | {best_str} | {worst_str} |")

    return "\n".join(lines)


def main():
    results_dir = Path("results")

    print("Loading round-robin results...")
    results = load_rr_results(results_dir)
    print(f"Loaded {len(results)} environments")

    print("\nAnalyzing profits and ranks...")
    profit_summary, rank_summary, overall = analyze_profits_and_ranks(results)

    print("\n" + "=" * 60)
    print("ROUND-ROBIN TOURNAMENT RESULTS (Santa Fe 1991 Traders)")
    print("=" * 60)

    print("\n### 2.3.1 Profit by Strategy (mean ± std per round)\n")
    print(format_profit_table(profit_summary))

    print("\n### 2.3.2 Rank by Environment (1=best)\n")
    print(format_rank_table(rank_summary))

    print("\n### 2.3.3 Tournament Summary\n")
    print(format_tournament_summary(rank_summary, overall))

    # Save JSON
    output = {
        "profit_summary": {
            env: {k: {kk: float(vv) for kk, vv in v.items()} for k, v in envdata.items()}
            for env, envdata in profit_summary.items()
        },
        "rank_summary": {
            env: {k: {kk: float(vv) for kk, vv in v.items()} for k, v in envdata.items()}
            for env, envdata in rank_summary.items()
        },
        "overall": {
            "avg_rank": {k: float(v) for k, v in overall["avg_rank"].items()},
            "wins": overall["wins"],
        },
    }

    with open(results_dir / "p2_rr_aggregated.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: results/p2_rr_aggregated.json")


if __name__ == "__main__":
    main()

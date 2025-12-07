#!/usr/bin/env python3
"""Aggregate ZIP round-robin results for comparison."""
import json
from pathlib import Path

import pandas as pd

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]


def aggregate_rr_results(result_prefix: str = "p2_rr_mixed"):
    """Aggregate round-robin results by strategy."""
    results_dir = Path("results")

    all_data = []

    for env in ENVS:
        result_dir = results_dir / f"{result_prefix}_{env}"
        results_file = result_dir / "results.csv"

        if not results_file.exists():
            print(f"Warning: {results_file} not found")
            continue

        df_env = pd.read_csv(results_file)

        # Sum profit by round and agent (across periods)
        # Remove trailing digit from agent_type (e.g., ZIC1 -> ZIC)
        df_env["strategy"] = df_env["agent_type"].str.replace(r"\d+$", "", regex=True)

        # Aggregate profit per round per agent
        round_profit = (
            df_env.groupby(["round", "agent_id", "strategy"])["period_profit"].sum().reset_index()
        )
        round_profit.rename(columns={"period_profit": "profit"}, inplace=True)

        # Add rank within each round
        round_profit["rank"] = round_profit.groupby("round")["profit"].rank(ascending=False)

        round_profit["env"] = env.upper()
        all_data.append(round_profit)

    if not all_data:
        print(f"No data found for {result_prefix}")
        return None

    df = pd.concat(all_data, ignore_index=True)

    # Aggregate by strategy and environment
    profit_stats = df.groupby(["strategy", "env"])["profit"].agg(["mean", "std"]).reset_index()
    rank_stats = df.groupby(["strategy", "env"])["rank"].agg(["mean", "std"]).reset_index()

    return df, profit_stats, rank_stats


def main():
    print("=" * 70)
    print("Aggregating Round-Robin Results")
    print("=" * 70)

    # Aggregate original Santa Fe results
    print("\n### Original Santa Fe 1991 (12 traders)")
    df_orig, profit_orig, rank_orig = aggregate_rr_results("p2_rr_mixed")

    # Aggregate ZIP-included results
    print("\n### With ZIP Added (13 traders)")
    df_zip, profit_zip, rank_zip = aggregate_rr_results("p2_rr_mixed_zip")

    # Print profit comparison tables
    print("\n" + "=" * 70)
    print("PROFIT COMPARISON (Mean ± Std)")
    print("=" * 70)

    if profit_orig is not None and profit_zip is not None:
        strategies = sorted(
            set(profit_orig["strategy"].unique()) | set(profit_zip["strategy"].unique())
        )

        # Print table header
        print("\n### Original (12 Santa Fe traders)")
        print("| Strategy |", " | ".join(ENVS[:5]), "|")
        print("|----------|", " | ".join(["----"] * 5), "|")

        for strat in strategies:
            row = [strat]
            for env in ENVS[:5]:
                mask = (profit_orig["strategy"] == strat) & (profit_orig["env"] == env.upper())
                if mask.any():
                    mean = profit_orig.loc[mask, "mean"].values[0]
                    std = profit_orig.loc[mask, "std"].values[0]
                    row.append(f"{mean:.0f}±{std:.0f}")
                else:
                    row.append("-")
            print("|", " | ".join(row), "|")

        print("\n### With ZIP Added (13 traders)")
        print("| Strategy |", " | ".join(ENVS[:5]), "|")
        print("|----------|", " | ".join(["----"] * 5), "|")

        strategies_zip = sorted(profit_zip["strategy"].unique())
        for strat in strategies_zip:
            row = [strat]
            for env in ENVS[:5]:
                mask = (profit_zip["strategy"] == strat) & (profit_zip["env"] == env.upper())
                if mask.any():
                    mean = profit_zip.loc[mask, "mean"].values[0]
                    std = profit_zip.loc[mask, "std"].values[0]
                    row.append(f"{mean:.0f}±{std:.0f}")
                else:
                    row.append("-")
            print("|", " | ".join(row), "|")

        # Print ZIP-specific performance
        print("\n### ZIP Performance in Mixed Market")
        print("| Environment | ZIP Profit | ZIP Rank |")
        print("|-------------|------------|----------|")
        for env in ENVS:
            p_mask = (profit_zip["strategy"] == "ZIP") & (profit_zip["env"] == env.upper())
            r_mask = (rank_zip["strategy"] == "ZIP") & (rank_zip["env"] == env.upper())
            if p_mask.any() and r_mask.any():
                p_mean = profit_zip.loc[p_mask, "mean"].values[0]
                p_std = profit_zip.loc[p_mask, "std"].values[0]
                r_mean = rank_zip.loc[r_mask, "mean"].values[0]
                print(f"| {env.upper()} | {p_mean:.0f}±{p_std:.0f} | {r_mean:.1f} |")

    # Save aggregated data
    if df_zip is not None:
        output = {
            "zip_profit_by_env": profit_zip.to_dict("records"),
            "zip_rank_by_env": rank_zip.to_dict("records"),
        }
        with open("results/p2_rr_zip_aggregated.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\nSaved: results/p2_rr_zip_aggregated.json")


if __name__ == "__main__":
    main()

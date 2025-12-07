#!/usr/bin/env python3
"""Get ZIP control metrics."""
from pathlib import Path

import pandas as pd

envs = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]

print("### ZIP Control (1 ZIP vs 7 ZIC)")
print("| Env | ZIP Profit | ZIC Profit | Ratio |")
print("|-----|------------|------------|-------|")
for env in envs:
    f = Path(f"results/p2_ctrl_zip_{env}/results.csv")
    if f.exists():
        df = pd.read_csv(f)
        df["strategy"] = df["agent_type"].str.replace(r"\d+$", "", regex=True)

        # Sum profit by round and agent
        round_profit = (
            df.groupby(["round", "agent_id", "strategy"])["period_profit"].sum().reset_index()
        )

        zip_profit = round_profit[round_profit["strategy"] == "ZIP"]["period_profit"].mean()
        zic_profit = round_profit[round_profit["strategy"] == "ZIC"]["period_profit"].mean()
        if zic_profit != 0:
            ratio = zip_profit / zic_profit
        else:
            ratio = float("inf")
        print(f"| {env.upper()} | {zip_profit:.0f} | {zic_profit:.0f} | {ratio:.2f} |")
    else:
        print(f"| {env.upper()} | - | - | - |")

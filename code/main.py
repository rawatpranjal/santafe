# main.py

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import os

from config import CONFIG
from auction import Auction

def print_role_strat(agg_dict):
    # Helper to pretty-print aggregator
    # agg_dict is the round's "role_strat_perf" dictionary
    #   keyed by (role, strategy), each with {profit, count}
    rows = []
    for (role, strat), val in agg_dict.items():
        total_p = val["profit"]
        count = val["count"]
        avg_p = total_p / count if count>0 else 0.0
        rows.append([role, strat, total_p, avg_p])

    # Convert to DataFrame for easy printing
    df = pd.DataFrame(rows, columns=["role","strategy","totalProfit","avgProfit"])
    print(df.to_string(index=False))

def main():
    os.makedirs("code/data", exist_ok=True)

    auction = Auction(CONFIG)
    auction.run_auction()

    # Load round-stats
    dfR = pd.DataFrame(auction.round_stats)

    # Just store the DataFrame for reference
    dfR.to_csv("code/data/round_stats.csv", index=False)
    print("Saved round_stats to code/data/round_stats.csv")

    # Print aggregator for first and last rounds
    first_round_agg = auction.round_stats[0]["role_strat_perf"]
    last_round_agg  = auction.round_stats[-1]["role_strat_perf"]

    print("\n=== Role-Strategy Data for Round 0 ===")
    print_role_strat(first_round_agg)

    print("\n=== Role-Strategy Data for Last Round ===")
    print_role_strat(last_round_agg)

    # Example plot
    plt.figure(figsize=(6,4))
    plt.plot(dfR["round"], dfR["market_efficiency"], label="Market Efficiency")
    plt.xlabel("Round")
    plt.ylabel("Efficiency")
    plt.title("Efficiency over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("code/data/efficiency.png", dpi=150)
    plt.close()
    print("Saved figure to code/data/efficiency.png")

if __name__ == "__main__":
    main()

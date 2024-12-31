import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import os

from config import CONFIG
from auction import Auction

def main():
    # Create output folder
    os.makedirs("data", exist_ok=True)

    auction = Auction(CONFIG)
    auction.run_auction()

    # Convert round-level stats to DataFrame
    dfR = pd.DataFrame(auction.round_stats)
    dfR.to_csv("data/round_stats.csv", index=False)
    print("Saved round_stats to data/round_stats.csv")

    # ==============================
    # 1) Print BOT-LEVEL data for round 0 and last round
    # ==============================
    def print_bot_details(bot_list):
        df = pd.DataFrame(bot_list, columns=["name","role","strategy","profit"])
        print(df.to_string(index=False))

    print("\n=== BOT-LEVEL DATA FOR ROUND 0 ===")
    df_bots_r0 = pd.DataFrame(auction.round_stats[0]["bot_details"])
    print_bot_details(df_bots_r0.to_dict(orient="records"))

    print("\n=== BOT-LEVEL DATA FOR LAST ROUND ===")
    df_bots_rLast = pd.DataFrame(auction.round_stats[-1]["bot_details"])
    print_bot_details(df_bots_rLast.to_dict(orient="records"))

    # ==============================
    # 2) Print MARKET PERFORMANCE for round 0 & last round
    # ==============================
    def print_market_perf(rstats):
        eff = rstats["market_efficiency"]
        diff_p = rstats["abs_diff_price"]
        diff_q = rstats["abs_diff_quantity"]
        print(f"  market_efficiency = {eff:.3f}")
        if diff_p is not None:
            print(f"  avg_price_diff    = {diff_p:.3f}")
        else:
            print(f"  avg_price_diff    = None")
        print(f"  avg_quant_diff    = {diff_q:.3f}")

    print("\n=== MARKET PERFORMANCE FOR ROUND 0 ===")
    print_market_perf(auction.round_stats[0])

    print("\n=== MARKET PERFORMANCE FOR LAST ROUND ===")
    print_market_perf(auction.round_stats[-1])

    # ==============================
    # 3) Rolling 100-round avgProfit by (role, strategy)
    # ==============================
    # Build a table: [round, role, strategy, avgProfit]
    rows = []
    for i in range(len(dfR)):
        rnum = dfR.loc[i, "round"]
        agg_dict = dfR.loc[i, "role_strat_perf"]
        for (role, strat), val in agg_dict.items():
            total_p = val["profit"]
            count = val["count"]
            avg_p = total_p / count if count>0 else 0.0
            rows.append([rnum, role, strat, avg_p])

    dfAgents = pd.DataFrame(rows, columns=["round","role","strategy","avgProfit"])

    # Pivot => each (role,strategy) is a column
    dfPivot = dfAgents.pivot_table(index="round",
                                   columns=["role","strategy"],
                                   values="avgProfit")

    # Rolling 100-round mean
    dfRolling = dfPivot.rolling(window=100, min_periods=1).mean()

    # Plot rolling avgProfit
    plt.figure(figsize=(8,6))
    for col in dfRolling.columns:
        label_str = f"{col[0]}-{col[1]}"
        plt.plot(dfRolling.index, dfRolling[col], label=label_str)
    plt.xlabel("Round")
    plt.ylabel("Avg Profit (100-round rolling mean)")
    plt.title("Agent Avg Profit Over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/avg_profit_by_agent.png", dpi=150)
    plt.close()
    print("Saved figure to data/avg_profit_by_agent.png")

    # ==============================
    # 4) Plot market efficiency
    # ==============================
    plt.figure(figsize=(6,4))
    plt.plot(dfR["round"], dfR["market_efficiency"], label="Market Efficiency")
    plt.xlabel("Round")
    plt.ylabel("Efficiency")
    plt.title("Efficiency over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/efficiency.png", dpi=150)
    plt.close()
    print("Saved figure to data/efficiency.png")


if __name__ == "__main__":
    main()

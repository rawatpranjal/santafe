# main.py

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np

from config import CONFIG
from auction import Auction

def main():
    auction = Auction(CONFIG)
    auction.run_auction()

    dfR = pd.DataFrame(auction.round_stats)
    print("\n=== Round-Level Stats ===")
    print(dfR.to_string(index=False))

    avg_eff = dfR["market_efficiency"].mean()
    avg_absdp = dfR["abs_diff_price"].mean()
    avg_absdq = dfR["abs_diff_quantity"].mean()

    print("\nAggregate over %d rounds:" % CONFIG["num_rounds"])
    print("  average market efficiency = %.3f" % avg_eff)
    print("  average |avgPrice - eqPrice| = %.4f" % avg_absdp)
    print("  average |actualTrades - eqQuantity| = %.4f" % avg_absdq)

    # OPTIONAL: if you want aggregator table:
    aggregator = auction.agg_strat_perf
    rows2=[]
    for k,v in aggregator.items():
         role, strat = k
         totalP = v["profit"]
         count = v["count"]
         avg_per_trader_per_round = (totalP / count) / float(CONFIG["num_rounds"])
         rows2.append([role, strat, totalP, avg_per_trader_per_round])
    dfStrat = pd.DataFrame(rows2, columns=["role","strategy","totalProfit","avgProfit_perTraderPerRound"])
    print("\n=== Aggregated Strategy-Role ===")
    print(dfStrat.to_string(index=False))

    dfR.to_csv("round_stats.csv", index=False)
    print("\nSaved round-level stats to 'round_stats.csv'")

if __name__=="__main__":
    main()

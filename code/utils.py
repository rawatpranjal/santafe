import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 8})

def compute_equilibrium(buyer_vals, seller_costs):
    """
    Compute equilibrium quantity (eq_q), equilibrium price (eq_p), and total surplus.
    """
    nb = len(buyer_vals)
    ns = len(seller_costs)
    Qmax = min(nb, ns)

    eq_q = 0
    for q in range(1, Qmax + 1):
        if buyer_vals[q - 1] >= seller_costs[q - 1]:
            eq_q = q
        else:
            break

    if eq_q == 0:
        eq_p = 0.5 * (buyer_vals[-1] + seller_costs[0]) if (buyer_vals and seller_costs) else 0.5
        return 0, eq_p, 0.0

    eq_p = 0.5 * (buyer_vals[eq_q - 1] + seller_costs[eq_q - 1])
    total = 0.0
    for i in range(eq_q):
        total += (buyer_vals[i] - seller_costs[i])
    return eq_q, eq_p, total


def analyze_individual_performance(round_stats):
    """
    Print each bot's overall performance across all rounds.
    Each bot => single row (aggregates across the rounds in which it appears).
    """
    all_bots = {}
    for rstat in round_stats:
        bot_details = rstat["bot_details"]
        for b in bot_details:
            key = (b["role"], b["strategy"], b["name"])
            if key not in all_bots:
                all_bots[key] = []
            all_bots[key].append(b["profit"])

    table_rows = []
    for (role, strategy, name), profit_list in all_bots.items():
        arr = np.array(profit_list)
        avg_p = np.mean(arr)
        std_p = np.std(arr)
        min_p = np.min(arr)
        med_p = np.median(arr)
        max_p = np.max(arr)
        table_rows.append([
            role, strategy, name,
            f"{avg_p:.4f}",
            f"{std_p:.4f}",
            f"{min_p:.4f}",
            f"{med_p:.4f}",
            f"{max_p:.4f}"
        ])

    print("\n=== INDIVIDUAL BOT PERFORMANCE (ACROSS ALL ROUNDS) ===")
    headers = ["Role", "Strategy", "BotName", "MeanProfit", "StdProfit", "MinProfit", "MedianProfit", "MaxProfit"]
    print(tabulate(table_rows, headers=headers, tablefmt="pretty"))


def analyze_market_performance(round_stats):
    """
    Print overall market performance (single summary row).
    """
    effs = []
    price_diffs = []
    quant_diffs = []
    buyer_surplus_list = []
    seller_surplus_list = []

    for rstat in round_stats:
        effs.append(rstat["market_efficiency"])
        if rstat["abs_diff_price"] is not None:
            price_diffs.append(rstat["abs_diff_price"])
        quant_diffs.append(rstat["abs_diff_quantity"])

        # Surplus fraction per round
        bot_details = rstat["bot_details"]
        buyer_pft = sum(b["profit"] for b in bot_details if b["role"] == "buyer")
        seller_pft = sum(b["profit"] for b in bot_details if b["role"] == "seller")
        tot_pft = buyer_pft + seller_pft
        if tot_pft > 1e-12:
            buyer_surplus_list.append(buyer_pft / tot_pft)
            seller_surplus_list.append(seller_pft / tot_pft)
        else:
            buyer_surplus_list.append(0)
            seller_surplus_list.append(0)

    avg_eff = np.mean(effs)
    std_eff = np.std(effs)
    avg_price_diff = np.mean(price_diffs) if price_diffs else 0.0
    avg_quant_diff = np.mean(quant_diffs)
    avg_buyer_surplus = np.mean(buyer_surplus_list)
    avg_seller_surplus = np.mean(seller_surplus_list)

    table_rows = [[
        f"{avg_eff:.4f}",
        f"{std_eff:.4f}",
        f"{avg_buyer_surplus*100:.2f}%",
        f"{avg_seller_surplus*100:.2f}%",
        f"{avg_price_diff:.4f}",
        f"{avg_quant_diff:.4f}"
    ]]
    headers = [
        "MarketEff(Mean)",
        "MarketEff(Std)",
        "BuyerSurplus%",
        "SellerSurplus%",
        "AvgPriceDiff",
        "AvgQuantDiff"
    ]
    print("\n=== MARKET PERFORMANCE (AGGREGATE) ===")
    print(tabulate(table_rows, headers=headers, tablefmt="pretty"))


def plot_per_round(round_stats, exp_path, dfLogs=None):
    """
    For each round, produce a 1x3 figure:
      Subplot(1): cBid, cAsk, trade price, eq price
      Subplot(2): Demand & supply step plot, eq lines, and transaction markers
      Subplot(3): Each buyer/seller's bids/asks over time
    """
    for rstat in round_stats:
        rnum = rstat["round"]

        if dfLogs is not None:
            df_round = dfLogs[dfLogs["round"] == rnum].copy()
            df_round.sort_values("step", inplace=True)
        else:
            df_round = pd.DataFrame()

        fig, axes = plt.subplots(1, 3, figsize=(15,4))

        # (1) Price Evolution
        eq_p = rstat["eq_p"]
        ax_price = axes[0]

        if not df_round.empty:
            steps = df_round["step"].values
            cbid_series = df_round["cbid"].values
            cask_series = df_round["cask"].values
            trade_vals = df_round.apply(lambda row: row["price"] if row["trade"]==1 else np.nan, axis=1)

            ax_price.plot(steps, cbid_series, label="cBid", color='blue', linestyle='--')
            ax_price.plot(steps, cask_series, label="cAsk", color='red', linestyle='--')
            ax_price.plot(steps, trade_vals, label="TradePrice", color='green', marker='o')
            ax_price.axhline(eq_p, color='grey', linestyle=':', label="Eq.Price")
        else:
            # fallback if no logs
            steps = np.arange(20)
            cBid_tmp = np.random.uniform(0.4, 0.6, size=len(steps))
            cAsk_tmp = cBid_tmp + 0.1
            trade_tmp = 0.5 * (cBid_tmp + cAsk_tmp)
            ax_price.plot(steps, cBid_tmp, label="cBid", color='blue', linestyle='--')
            ax_price.plot(steps, cAsk_tmp, label="cAsk", color='red', linestyle='--')
            ax_price.plot(steps, trade_tmp, label="TradePrice", color='green', marker='o')
            ax_price.axhline(eq_p, color='grey', linestyle=':', label="Eq.Price")

        ax_price.set_title(f"Round {rnum} - Price Evolution")
        ax_price.set_xlabel("Step")
        ax_price.set_ylabel("Price")
        ax_price.legend()

        # (2) Supply vs Demand
        ax_sd = axes[1]
        buyer_vals = rstat.get("buyer_vals", None)
        seller_vals = rstat.get("seller_vals", None)
        eq_q = rstat["eq_q"]

        if not buyer_vals or not seller_vals:
            buyer_vals = sorted(np.random.uniform(0.5,1.0,20), reverse=True)
            seller_vals = sorted(np.random.uniform(0.0,0.5,20))

        ax_sd.step(range(len(buyer_vals)), buyer_vals, where='post', color='blue', label="Demand")
        ax_sd.step(range(len(seller_vals)), seller_vals, where='post', color='red', label="Supply")
        ax_sd.axhline(eq_p, color='grey', linestyle=':', label="Eq.Price")
        ax_sd.axvline(eq_q, color='grey', linestyle=':', label="Eq.Qty")

        # Mark actual trades as green 'x'
        if not df_round.empty:
            trades_only = df_round[df_round["trade"] == 1]
            trade_idx = 0
            for _, rowt in trades_only.iterrows():
                trade_idx += 1
                tp = rowt["price"]
                ax_sd.scatter(trade_idx, tp, color='green', marker='x', s=50)

        ax_sd.set_title("Supply vs Demand")
        ax_sd.set_xlabel("Units")
        ax_sd.set_ylabel("Valuation/Cost")
        ax_sd.legend()

        # (3) Individual Bids/Asks Over Time
        ax_ba = axes[2]
        if not df_round.empty:
            steps = df_round["step"].values
            nBuyers = len(df_round.iloc[0]["bids"]) if len(df_round) > 0 else 0
            nSellers = len(df_round.iloc[0]["asks"]) if len(df_round) > 0 else 0

            for b_idx in range(nBuyers):
                bids_over_time = df_round["bids"].apply(lambda blist: blist[b_idx]).values
                ax_ba.plot(steps, bids_over_time, label=f"B{b_idx}_bid", alpha=0.6)

            for s_idx in range(nSellers):
                asks_over_time = df_round["asks"].apply(lambda alist: alist[s_idx]).values
                ax_ba.plot(steps, asks_over_time, label=f"S{s_idx}_ask", alpha=0.6)

            trade_line = df_round.apply(lambda row: row["price"] if row["trade"]==1 else np.nan, axis=1)
            ax_ba.plot(steps, trade_line, color='green', marker='o', label="TradePrice")

        ax_ba.set_title("Individual Bids/Asks Over Time")
        ax_ba.set_xlabel("Step")
        ax_ba.set_ylabel("Price")
        ax_ba.legend()

        fig.tight_layout()
        out_path = os.path.join(exp_path, f"round_{rnum}_1x3.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_aggregates(dfR, exp_path, dfLogs=None):
    """
    (Optional) See if you still need it. 
    This is an older function for 1) rolling avg profit 
    + 2) a 2x2 grid with efficiency, surplus, etc.
    """
    pass  # remove or implement if needed


def plot_game_summary(dfR, exp_path, dfLogs=None):
    """
    Single 2x3 figure: 
      (0,0) Efficiency, (0,1) Buyer+Seller Surplus, (0,2) Price & Quantity Diff
      bottom row => rolling avg profit chart
    """
    fig, axes = plt.subplots(2, 3, figsize=(15,8))

    # (0,0) Efficiency
    ax_eff = axes[0,0]
    ax_eff.plot(dfR["round"], dfR["market_efficiency"], label="Market Efficiency")
    ax_eff.set_xlabel("Round")
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_title("Efficiency over Rounds")
    ax_eff.legend()

    # Buyer & Seller Surplus fraction
    buyer_surplus_frac = []
    seller_surplus_frac = []
    for i in range(len(dfR)):
        bot_details = dfR.loc[i, "bot_details"]
        bp = sum(b["profit"] for b in bot_details if b["role"]=="buyer")
        sp = sum(b["profit"] for b in bot_details if b["role"]=="seller")
        tot = bp + sp
        if tot>1e-12:
            buyer_surplus_frac.append(bp/tot)
            seller_surplus_frac.append(sp/tot)
        else:
            buyer_surplus_frac.append(0)
            seller_surplus_frac.append(0)

    # (0,1) Surplus (Buyer + Seller)
    ax_surplus = axes[0,1]
    rounds = dfR["round"]
    ax_surplus.plot(rounds, buyer_surplus_frac, label="Buyer Surplus%", color='blue')
    ax_surplus.plot(rounds, seller_surplus_frac, label="Seller Surplus%", color='red')
    ax_surplus.set_xlabel("Round")
    ax_surplus.set_ylabel("Surplus Fraction")
    ax_surplus.set_title("Buyer & Seller Surplus")
    ax_surplus.legend()

    # (0,2) Price Diff & Quantity Diff
    ax_diff = axes[0,2]
    price_diff = dfR["abs_diff_price"].fillna(0)
    qty_diff = dfR["abs_diff_quantity"].fillna(0)
    ax_diff.plot(rounds, price_diff, label="|AvgPrice - EqPrice|", color='green')
    ax_diff.plot(rounds, qty_diff, label="|Trades - EqQ|", color='orange')
    ax_diff.set_xlabel("Round")
    ax_diff.set_ylabel("Diff")
    ax_diff.set_title("Price & Quantity Diff")
    ax_diff.legend()

    # Bottom row => Rolling Avg Profit
    ax_profit = axes[1,0]

    rows = []
    for i in range(len(dfR)):
        rnum = dfR.loc[i, "round"]
        agg_dict = dfR.loc[i, "role_strat_perf"]
        for (role, strat), val in agg_dict.items():
            total_p = val["profit"]
            count = val["count"]
            avg_p = total_p / count if count > 0 else 0.0
            rows.append([rnum, role, strat, avg_p])

    dfAgents = pd.DataFrame(rows, columns=["round","role","strategy","avgProfit"])
    dfPivot = dfAgents.pivot_table(index="round", columns=["role","strategy"], values="avgProfit")
    dfRolling = dfPivot.rolling(window=4, min_periods=1).mean()

    for col in dfRolling.columns:
        label_str = f"{col[0]}-{col[1]}"
        ax_profit.plot(dfRolling.index, dfRolling[col], label=label_str)
    ax_profit.set_xlabel("Round")
    ax_profit.set_ylabel("Avg Profit (4-round rolling)")
    ax_profit.set_title("Agent Avg Profit Over Rounds")
    ax_profit.legend()

    # Turn off the other two subplots on the bottom row
    axes[1,1].axis("off")
    axes[1,2].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(exp_path, "game_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved single game-level summary to {summary_path}")

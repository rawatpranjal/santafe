# auction.py

import random
import numpy as np

from utils import compute_equilibrium
from traders.registry import get_trader_class  # <-- import the registry's factory

class Auction:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config["num_rounds"]
        self.num_periods = config.get("num_periods", 1)
        self.num_steps = config["num_steps"]

        self.round_stats = []
        self.all_step_logs = []

        # We keep track of best bid/ask each step
        self.current_bid = None
        self.current_ask = None

    def run_auction(self):
        for r in range(self.num_rounds):
            buyers, sellers = self.create_traders_for_round(r)

            buyer_vals = []
            for b in buyers:
                buyer_vals.extend(b.private_values)
            buyer_vals.sort(reverse=True)

            seller_costs = []
            for s in sellers:
                seller_costs.extend(s.private_values)
            seller_costs.sort()

            eq_q, eq_p, eq_surplus = compute_equilibrium(buyer_vals, seller_costs)
            rstats = self.run_round(r, buyers, sellers, eq_q, eq_p, eq_surplus)
            rstats["buyer_vals"] = buyer_vals
            rstats["seller_vals"] = seller_costs
            self.round_stats.append(rstats)

            # RL finishing calls
            for b in buyers:
                if hasattr(b, "agent") and hasattr(b.agent, "finish_round"):
                    b.agent.finish_round()
            for s in sellers:
                if hasattr(s, "agent") and hasattr(s.agent, "finish_round"):
                    s.agent.finish_round()

    def create_traders_for_round(self, r):
        rng = random.Random(1337 + r*777)
        nT = self.config["num_tokens"]

        buyers = []
        for i, spec in enumerate(self.config["buyers"]):
            vals = sorted([
                rng.uniform(
                    self.config["buyer_valuation_min"],
                    self.config["buyer_valuation_max"]
                ) for _ in range(nT)], reverse=True)
            name = f"B{i}"
            TraderCls = get_trader_class(spec["type"], True)
            init_args = spec.get("init_args", {})
            t = TraderCls(name, True, vals, **init_args)
            buyers.append(t)

        sellers = []
        for j, spec in enumerate(self.config["sellers"]):
            vals = sorted([
                rng.uniform(
                    self.config["seller_cost_min"],
                    self.config["seller_cost_max"]
                ) for _ in range(nT)])
            name = f"S{j}"
            TraderCls = get_trader_class(spec["type"], False)
            init_args = spec.get("init_args", {})
            t = TraderCls(name, False, vals, **init_args)
            sellers.append(t)

        return buyers, sellers

    def run_round(self, r, buyers, sellers, eq_q, eq_p, eq_surplus):
        self.current_bid = None
        self.current_ask = None

        for t in buyers + sellers:
            t.reset_for_new_period(self.num_steps, r, 0)

        nTrades = 0
        trades_price = []

        for st in range(self.num_steps):
            trade_price = self.run_step(r, 0, st, buyers, sellers)
            if trade_price is not None:
                nTrades += 1
                trades_price.append(trade_price)

        tot_bprofit = sum(b.profit for b in buyers)
        tot_sprofit = sum(s.profit for s in sellers)
        tot_profit = tot_bprofit + tot_sprofit
        efficiency = tot_profit / eq_surplus if eq_surplus>1e-12 else 0.0

        avg_p, adiff_p = (None, None)
        if trades_price:
            avg_p = np.mean(trades_price)
            adiff_p = abs(avg_p - eq_p)
        adiff_q = abs(nTrades - eq_q)

        # aggregator
        round_strat = {}
        for b in buyers:
            k = ("buyer", b.strategy)
            if k not in round_strat:
                round_strat[k] = {"profit": 0.0, "count": 0}
            round_strat[k]["profit"] += b.profit
            round_strat[k]["count"] += 1

        for s in sellers:
            k = ("seller", s.strategy)
            if k not in round_strat:
                round_strat[k] = {"profit": 0.0, "count": 0}
            round_strat[k]["profit"] += s.profit
            round_strat[k]["count"] += 1

        # bot-level
        bot_details = []
        for b in buyers:
            bot_details.append({
                "name": b.name,
                "role": "buyer",
                "strategy": b.strategy,
                "profit": b.profit
            })
        for s in sellers:
            bot_details.append({
                "name": s.name,
                "role": "seller",
                "strategy": s.strategy,
                "profit": s.profit
            })

        return {
            "round": r,
            "eq_q": eq_q,
            "eq_p": eq_p,
            "eq_surplus": eq_surplus,
            "actual_trades": nTrades,
            "actual_total_profit": tot_profit,
            "market_efficiency": efficiency,
            "avg_price": avg_p,
            "abs_diff_price": adiff_p,
            "abs_diff_quantity": adiff_q,
            "role_strat_perf": round_strat,
            "bot_details": bot_details
        }

    def run_step(self, r, p, st, buyers, sellers):
        for t in buyers + sellers:
            t.current_step = st

        # We'll gather new bids/asks in lists
        bid_list = [None]*len(buyers)
        ask_list = [None]*len(sellers)

        c_bid_val = self.current_bid[0] if self.current_bid else None
        c_ask_val = self.current_ask[0] if self.current_ask else None

        # Collect new bids
        for i, b in enumerate(buyers):
            off = b.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                price_2dec = round(off[0], 2)
                bid_list[i] = price_2dec

        # Collect new asks
        for j, s in enumerate(sellers):
            off = s.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                price_2dec = round(off[0], 2)
                ask_list[j] = price_2dec

        # Recompute best bid
        valid_bids = [(val, buyers[i]) for i,val in enumerate(bid_list) if val is not None]
        if valid_bids:
            best_bid = max(valid_bids, key=lambda x: x[0])
            self.current_bid = best_bid
        else:
            self.current_bid = None

        # Recompute best ask
        valid_asks = [(val, sellers[j]) for j,val in enumerate(ask_list) if val is not None]
        if valid_asks:
            best_ask = min(valid_asks, key=lambda x: x[0])
            self.current_ask = best_ask
        else:
            self.current_ask = None

        # Attempt trade
        trade_price = None
        trade_happened = 0
        if self.current_bid and self.current_ask:
            bid_val, btrader = self.current_bid
            ask_val, strader = self.current_ask
            if btrader.decide_to_buy(ask_val) and strader.decide_to_sell(bid_val):
                tprice = 0.5*(bid_val + ask_val)
                trade_price = round(tprice, 2)
                btrader.transact(trade_price)
                strader.transact(trade_price)
                self.current_bid = None
                self.current_ask = None
                trade_happened = 1

        # If no trade => keep best bid/ask from above
        cBidFinal = None
        cAskFinal = None
        if self.current_bid:
            cBidFinal = round(self.current_bid[0],2)
        if self.current_ask:
            cAskFinal = round(self.current_ask[0],2)

        row = {
            "round": r,
            "period": p,
            "step": st,
            "bids": bid_list,   # each buyer's price or None
            "asks": ask_list,   # each seller's price or None
            "cbid": cBidFinal,
            "cask": cAskFinal,
            "trade": trade_happened,
            "price": trade_price,
            "bprofits": [round(b.profit,2) for b in buyers],
            "sprofits": [round(s.profit,2) for s in sellers]
        }
        self.all_step_logs.append(row)

        return trade_price

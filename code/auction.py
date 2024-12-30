# auction.py

import random
import numpy as np

from utils import compute_equilibrium
from traders.zic import RandomBuyer, RandomSeller
from traders.kaplan import ImprovedKaplanBuyer, ImprovedKaplanSeller
from traders.gd import GDBuyer, GDSeller
from traders.zip import ZipBuyer, ZipSeller  # <-- IMPORTANT

class Auction:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config["num_rounds"]
        self.num_periods = config["num_periods"]
        self.num_steps = config["num_steps"]

        self.logs = []
        self.round_stats = []

        # If you still had an aggregator, you can keep it or remove it entirely:
        self.agg_strat_perf = {}

        # current best bid & ask each step
        self.current_bid = None
        self.current_ask = None

    def run_auction(self):
        for r in range(self.num_rounds):
            buyers, sellers = self.create_traders_for_round(r)

            # equilibrium
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
            self.round_stats.append(rstats)

            # If you had aggregator logic, you can keep or remove:
            for b in buyers:
                key = ("buyer", b.strategy)
                if key not in self.agg_strat_perf:
                    self.agg_strat_perf[key] = {"profit":0.0, "count":0}
                self.agg_strat_perf[key]["profit"] += b.profit
                self.agg_strat_perf[key]["count"] += 1

            for s in sellers:
                key = ("seller", s.strategy)
                if key not in self.agg_strat_perf:
                    self.agg_strat_perf[key] = {"profit":0.0, "count":0}
                self.agg_strat_perf[key]["profit"] += s.profit
                self.agg_strat_perf[key]["count"] += 1

    def create_traders_for_round(self, r):
        rng = random.Random(1337 + r*777)
        nT = self.config["num_tokens"]

        # Buyers
        buyers = []
        for i, spec in enumerate(self.config["buyers"]):
            vals = sorted(
                [rng.uniform(self.config["buyer_valuation_min"],
                             self.config["buyer_valuation_max"])
                 for _ in range(nT)],
                reverse=True
            )
            name = f"B{i}_r{r}"

            if spec["type"] == "random":
                t = RandomBuyer(name, True, vals)
            elif spec["type"] == "kaplan":
                t = ImprovedKaplanBuyer(name, True, vals, margin=self.config["sniper_margin"])
            elif spec["type"] == "gdbuyer":
                t = GDBuyer(name, True, vals, grid_size=10)
            elif spec["type"] == "zipbuyer":  # <-- ADDED
                t = ZipBuyer(name, True, vals, margin_init=0.05)
            else:
                raise ValueError("Unknown buyer type:", spec["type"])
            buyers.append(t)

        # Sellers
        sellers = []
        for j, spec in enumerate(self.config["sellers"]):
            vals = sorted(
                [rng.uniform(self.config["seller_cost_min"],
                             self.config["seller_cost_max"])
                 for _ in range(nT)]
            )
            name = f"S{j}_r{r}"

            if spec["type"] == "random":
                t = RandomSeller(name, False, vals)
            elif spec["type"] == "kaplan":
                t = ImprovedKaplanSeller(name, False, vals, margin=self.config["sniper_margin"])
            elif spec["type"] == "gdseller":
                t = GDSeller(name, False, vals, grid_size=10)
            elif spec["type"] == "zipseller":  # <-- ADDED
                t = ZipSeller(name, False, vals, margin_init=0.05)
            else:
                raise ValueError("Unknown seller type:", spec["type"])
            sellers.append(t)

        return buyers, sellers

    def run_round(self, r, buyers, sellers, eq_q, eq_p, eq_surplus):
        self.current_bid = None
        self.current_ask = None

        for t in buyers + sellers:
            t.reset_for_new_period(self.num_steps, r, 0)

        nTrades=0
        trades_price=[]
        for st in range(self.num_steps):
            out = self.run_step(r, 0, st, buyers, sellers)
            if out is not None:
                nTrades += 1
                trades_price.append(out)

        tot_bprofit = sum(b.profit for b in buyers)
        tot_sprofit = sum(s.profit for s in sellers)
        tot_profit = tot_bprofit + tot_sprofit
        eff = tot_profit/eq_surplus if eq_surplus>1e-12 else 0.0

        if len(trades_price)>0:
            avg_p = np.mean(trades_price)
            adiff_p = abs(avg_p - eq_p)
        else:
            avg_p = None
            adiff_p = None
        adiff_q = abs(nTrades - eq_q)

        return {
            "round": r,
            "eq_q": eq_q,
            "eq_p": eq_p,
            "eq_surplus": eq_surplus,
            "actual_trades": nTrades,
            "actual_total_profit": tot_profit,
            "market_efficiency": eff,
            "avg_price": avg_p,
            "abs_diff_price": adiff_p,
            "abs_diff_quantity": adiff_q
        }

    def run_step(self, r, p, st, buyers, sellers):
        for t in buyers+sellers:
            t.current_step = st

        c_bid_val = self.current_bid[0] if self.current_bid else None
        c_ask_val = self.current_ask[0] if self.current_ask else None

        new_bids=[]
        for b in buyers:
            off = b.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                new_bids.append(off)

        new_asks=[]
        for s in sellers:
            off = s.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                new_asks.append(off)

        if new_bids:
            best_new_bid = max(new_bids, key=lambda x: x[0])
            if (self.current_bid is None) or (best_new_bid[0] > self.current_bid[0]):
                self.current_bid = best_new_bid
        if new_asks:
            best_new_ask = min(new_asks, key=lambda x: x[0])
            if (self.current_ask is None) or (best_new_ask[0] < self.current_ask[0]):
                self.current_ask = best_new_ask

        trade_price = None
        if self.current_bid and self.current_ask:
            bid_val, btrader = self.current_bid
            ask_val, strader = self.current_ask
            if btrader.decide_to_buy(ask_val) and strader.decide_to_sell(bid_val):
                trade_price = 0.5*(bid_val+ask_val)
                btrader.transact(trade_price)
                strader.transact(trade_price)

                # if GD or ZIP or any approach uses .observe_trade():
                if hasattr(btrader, "observe_trade"):
                    btrader.observe_trade(trade_price)
                if hasattr(strader, "observe_trade"):
                    strader.observe_trade(trade_price)

                self.current_bid = None
                self.current_ask = None

        return trade_price

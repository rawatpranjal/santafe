# auction.py

import random
import numpy as np

from utils import compute_equilibrium
from traders.registry import get_trader_class  # <-- import the registry's factory

class Auction:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config["num_rounds"]
        self.num_periods = config["num_periods"]
        self.num_steps = config["num_steps"]

        self.round_stats = []

        # current best bid & ask
        self.current_bid = None
        self.current_ask = None

    def run_auction(self):
        """
        Main entry: run all rounds.
        """
        for r in range(self.num_rounds):
            buyers, sellers = self.create_traders_for_round(r)

            # gather valuations
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

            # If PPO or other custom finishing
            for b in buyers:
                if hasattr(b, "agent") and hasattr(b.agent, "finish_round"):
                    b.agent.finish_round()
            for s in sellers:
                if hasattr(s, "agent") and hasattr(s.agent, "finish_round"):
                    s.agent.finish_round()

    def create_traders_for_round(self, r):
        """
        Uses registry to create each buyer/seller from config specs.
        No direct references to custom classes.
        """
        rng = random.Random(1337 + r*777)
        nT = self.config["num_tokens"]

        # Buyers
        buyers = []
        for i, spec in enumerate(self.config["buyers"]):
            # generate valuations
            vals = sorted(
                [
                    rng.uniform(self.config["buyer_valuation_min"],
                                self.config["buyer_valuation_max"])
                    for _ in range(nT)
                ],
                reverse=True
            )
            name = f"B{i}_r{r}"

            TraderCls = get_trader_class(spec["type"], True)
            # If you have extra constructor args, pass them in
            init_args = spec.get("init_args", {})
            t = TraderCls(name, True, vals, **init_args)
            buyers.append(t)

        # Sellers
        sellers = []
        for j, spec in enumerate(self.config["sellers"]):
            vals = sorted(
                [
                    rng.uniform(self.config["seller_cost_min"],
                                self.config["seller_cost_max"])
                    for _ in range(nT)
                ]
            )
            name = f"S{j}_r{r}"

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
            key = ("buyer", b.strategy)
            if key not in round_strat:
                round_strat[key] = {"profit": 0.0, "count": 0}
            round_strat[key]["profit"] += b.profit
            round_strat[key]["count"] += 1

        for s in sellers:
            key = ("seller", s.strategy)
            if key not in round_strat:
                round_strat[key] = {"profit": 0.0, "count": 0}
            round_strat[key]["profit"] += s.profit
            round_strat[key]["count"] += 1

        # bot-level (optional)
        bot_details = []
        for b in buyers:
            bot_details.append({
                "name": b.name, "role":"buyer",
                "strategy": b.strategy, "profit": b.profit
            })
        for s in sellers:
            bot_details.append({
                "name": s.name, "role":"seller",
                "strategy": s.strategy, "profit": s.profit
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

        c_bid_val = self.current_bid[0] if self.current_bid else None
        c_ask_val = self.current_ask[0] if self.current_ask else None

        new_bids = []
        for b in buyers:
            off = b.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                new_bids.append(off)

        new_asks = []
        for s in sellers:
            off = s.make_bid_or_ask(c_bid_val, c_ask_val, 0, 1, 0.0, 1.0)
            if off is not None:
                new_asks.append(off)

        # update best bid
        if new_bids:
            best_new_bid = max(new_bids, key=lambda x: x[0])
            if (self.current_bid is None) or (best_new_bid[0] > self.current_bid[0]):
                self.current_bid = best_new_bid

        # update best ask
        if new_asks:
            best_new_ask = min(new_asks, key=lambda x: x[0])
            if (self.current_ask is None) or (best_new_ask[0] < self.current_ask[0]):
                self.current_ask = best_new_ask

        # check for trade
        trade_price = None
        if self.current_bid and self.current_ask:
            bid_val, btrader = self.current_bid
            ask_val, strader = self.current_ask
            if btrader.decide_to_buy(ask_val) and strader.decide_to_sell(bid_val):
                trade_price = 0.5*(bid_val + ask_val)
                reward_b = btrader.transact(trade_price)
                reward_s = strader.transact(trade_price)
                if hasattr(btrader, "update_trade_stats"):
                    btrader.update_trade_stats(trade_price)
                if hasattr(strader, "update_trade_stats"):
                    strader.update_trade_stats(trade_price)
                if hasattr(btrader, "update_after_trade"):
                    btrader.update_after_trade(reward_b)
                if hasattr(strader, "update_after_trade"):
                    strader.update_after_trade(reward_s)

                self.current_bid = None
                self.current_ask = None

        return trade_price

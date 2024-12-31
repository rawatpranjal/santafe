# auction.py

import random
import numpy as np

from utils import compute_equilibrium
from traders.zic import RandomBuyer, RandomSeller
from traders.gd import GDBuyer, GDSeller
from traders.zip import ZipBuyer, ZipSeller
from traders.kaplan import KaplanBuyer, KaplanSeller
from traders.ppo import PPOBuyer, PPOSeller  # <-- PPO import

class Auction:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config["num_rounds"]
        self.num_periods = config["num_periods"]
        self.num_steps = config["num_steps"]

        self.logs = []
        self.round_stats = []

        # current best bid & ask for the step
        self.current_bid = None
        self.current_ask = None

    def run_auction(self):
        for r in range(self.num_rounds):
            buyers, sellers = self.create_traders_for_round(r)

            # For reference: compute equilibrium for fresh valuations
            buyer_vals = []
            for b in buyers:
                buyer_vals.extend(b.private_values)
            buyer_vals.sort(reverse=True)

            seller_costs = []
            for s in sellers:
                seller_costs.extend(s.private_values)
            seller_costs.sort()

            eq_q, eq_p, eq_surplus = compute_equilibrium(buyer_vals, seller_costs)

            # Run the entire round
            rstats = self.run_round(r, buyers, sellers, eq_q, eq_p, eq_surplus)
            self.round_stats.append(rstats)

            # Let PPO agents update after each round
            for b in buyers:
                if hasattr(b, "agent"):
                    b.agent.update_policy()
            for s in sellers:
                if hasattr(s, "agent"):
                    s.agent.update_policy()

    def create_traders_for_round(self, r):
        """
        Create buyers & sellers for round r, using config specs.
        """
        rng = random.Random(1337 + r * 777)
        nT = self.config["num_tokens"]

        # --- BUYERS ---
        buyers = []
        for i, spec in enumerate(self.config["buyers"]):
            # generate private valuations
            vals = sorted(
                [
                    rng.uniform(
                        self.config["buyer_valuation_min"],
                        self.config["buyer_valuation_max"]
                    )
                    for _ in range(nT)
                ],
                reverse=True
            )
            name = f"B{i}_r{r}"

            if spec["type"] == "random":
                t = RandomBuyer(name, True, vals)
            elif spec["type"] == "kaplan":
                t = KaplanBuyer(name, True, vals)
            elif spec["type"] == "gdbuyer":
                t = GDBuyer(name, True, vals, grid_size=10)
            elif spec["type"] == "zipbuyer":
                t = ZipBuyer(name, True, vals, margin_init=0.05)
            elif spec["type"] == "ppobuyer":
                t = PPOBuyer(name, True, vals)
            else:
                raise ValueError("Unknown buyer type:", spec["type"])
            buyers.append(t)

        # --- SELLERS ---
        sellers = []
        for j, spec in enumerate(self.config["sellers"]):
            # generate private costs
            vals = sorted(
                [
                    rng.uniform(
                        self.config["seller_cost_min"],
                        self.config["seller_cost_max"]
                    )
                    for _ in range(nT)
                ]
            )
            name = f"S{j}_r{r}"

            if spec["type"] == "random":
                t = RandomSeller(name, False, vals)
            elif spec["type"] == "kaplan":
                t = KaplanSeller(name, False, vals)
            elif spec["type"] == "gdseller":
                t = GDSeller(name, False, vals, grid_size=10)
            elif spec["type"] == "zipseller":
                t = ZipSeller(name, False, vals, margin_init=0.05)
            elif spec["type"] == "pposeller":
                t = PPOSeller(name, False, vals)
            else:
                raise ValueError("Unknown seller type:", spec["type"])
            sellers.append(t)

        return buyers, sellers

    def run_round(self, r, buyers, sellers, eq_q, eq_p, eq_surplus):
        """
        Runs a single round (with exactly self.num_periods, typically 1).
        - Resets current_bid/current_ask to None
        - Each step calls run_step, tries to match trades
        - At end, collects stats & per-round aggregator
        """
        self.current_bid = None
        self.current_ask = None

        # single period for simplicity
        for t in buyers + sellers:
            t.reset_for_new_period(self.num_steps, r, 0)

        nTrades = 0
        trades_price = []

        for st in range(self.num_steps):
            trade_price = self.run_step(r, 0, st, buyers, sellers)
            if trade_price is not None:
                nTrades += 1
                trades_price.append(trade_price)

        # compute round-level stats
        tot_bprofit = sum(b.profit for b in buyers)
        tot_sprofit = sum(s.profit for s in sellers)
        tot_profit = tot_bprofit + tot_sprofit
        efficiency = tot_profit / eq_surplus if eq_surplus > 1e-12 else 0.0

        if trades_price:
            avg_p = np.mean(trades_price)
            adiff_p = abs(avg_p - eq_p)
        else:
            avg_p = None
            adiff_p = None
        adiff_q = abs(nTrades - eq_q)

        # -- Build a per-round role-strategy aggregator --
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
            "role_strat_perf": round_strat  # store aggregator for this round
        }

    def run_step(self, r, p, st, buyers, sellers):
        """
        Each step, each trader can post a new bid or ask.
        If bestBid >= bestAsk => a deal is made at midpoint.
        """
        for t in buyers + sellers:
            t.current_step = st

        # gather new bids
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

        trade_price = None
        if self.current_bid and self.current_ask:
            bid_val, btrader = self.current_bid
            ask_val, strader = self.current_ask

            # if both accept => trade
            if btrader.decide_to_buy(ask_val) and strader.decide_to_sell(bid_val):
                trade_price = 0.5 * (bid_val + ask_val)
                reward_b = btrader.transact(trade_price)
                reward_s = strader.transact(trade_price)

                # if GD or Kaplan or ZIP uses .observe_trade / .update_trade_stats
                if hasattr(btrader, "update_trade_stats"):
                    btrader.update_trade_stats(trade_price)
                if hasattr(strader, "update_trade_stats"):
                    strader.update_trade_stats(trade_price)

                # if PPO => update with reward
                if hasattr(btrader, "update_after_trade"):
                    btrader.update_after_trade(reward_b)
                if hasattr(strader, "update_after_trade"):
                    strader.update_after_trade(reward_s)

                # reset current quotes
                self.current_bid = None
                self.current_ask = None

        return trade_price

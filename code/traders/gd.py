# traders/gd.py

import random
from .base import BaseTrader

class GDBuyer(BaseTrader):
    def __init__(self, name, is_buyer, private_values, grid_size=10):
        super().__init__(name, is_buyer, private_values, "gd-buyer")
        self.grid_size = grid_size
        self.transaction_history = []

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.transaction_history.clear()  # fresh for new round

    def observe_trade(self, price):
        self.transaction_history.append(price)

    def probability_accepted(self, price):
        """
        q(price): fraction of trades <= price for buyer
        """
        if not self.transaction_history:
            return 0.0
        count = sum(1 for p in self.transaction_history if p <= price)
        return count / float(len(self.transaction_history))

    def expected_profit(self, price, limit):
        payoff = limit - price
        if payoff<=0: 
            return 0.0
        q = self.probability_accepted(price)
        return q*payoff

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        limit = self.next_token_value()
        if limit is None:
            return None

        # If no data => random conservative
        if not self.transaction_history:
            fallback_bid = random.uniform(0.0, limit)
            return (fallback_bid, self)

        # We'll build a small grid from 0..limit, e.g. grid_size=10 => 11 points
        candidate_prices = []
        step = max(1, self.grid_size)
        for i in range(step+1):
            p = limit*(i/step)
            candidate_prices.append(p)

        best_p = None
        bestE = -1.0
        for cp in candidate_prices:
            Ecp = self.expected_profit(cp, limit)
            if Ecp>bestE:
                bestE=Ecp
                best_p=cp

        return (best_p, self)

    def decide_to_buy(self, best_ask):
        val=self.next_token_value()
        return (val is not None) and (val>=best_ask)

    def decide_to_sell(self, best_bid):
        return False


class GDSeller(BaseTrader):
    def __init__(self, name, is_buyer, private_values, grid_size=10):
        super().__init__(name, is_buyer, private_values, "gd-seller")
        self.grid_size = grid_size
        self.transaction_history = []

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.transaction_history.clear()

    def observe_trade(self, price):
        self.transaction_history.append(price)

    def probability_accepted(self, price):
        """
        q(price): fraction of trades >= price for seller
        """
        if not self.transaction_history:
            return 0.0
        count = sum(1 for p in self.transaction_history if p >= price)
        return count / float(len(self.transaction_history))

    def expected_profit(self, price, cost):
        payoff = price - cost
        if payoff<=0:
            return 0.0
        q = self.probability_accepted(price)
        return q*payoff

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None

        # If no data => random conservative
        if not self.transaction_history:
            fallback_ask = random.uniform(cost, max_price)
            return (fallback_ask, self)

        # small grid from cost..max_price
        candidate_prices = []
        step = max(1, self.grid_size)
        for i in range(step+1):
            p = cost + (max_price-cost)*(i/step)
            candidate_prices.append(p)

        best_p = None
        bestE = -1.0
        for cp in candidate_prices:
            Ecp = self.expected_profit(cp, cost)
            if Ecp>bestE:
                bestE=Ecp
                best_p=cp

        return (best_p, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost=self.next_token_value()
        return (cost is not None) and (best_bid>=cost)

import random
from .base import BaseTrader

class KaplanBuyer(BaseTrader):
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="kaplan-buyer")
        self.min_price = 1.0   # track min trade so far
        self.max_price = 0.0   # track max trade so far
        self.sum_prices = 0.0
        self.num_trades = 0

    def update_trade_stats(self, trade_price):
        # track stats
        self.min_price = min(self.min_price, trade_price)
        self.max_price = max(self.max_price, trade_price)
        self.sum_prices += trade_price
        self.num_trades += 1

    def average_price(self):
        return (self.sum_prices / self.num_trades) if self.num_trades>0 else 0.5

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None

        # if no best_bid => fallback
        if c_bid is None or c_bid<=0:
            # e.g. pick half limit or small number like 0.01
            fallback = min(0.99*val, 0.5*val)
            return (fallback, self)

        # time is short => jump in
        time_left = (self.total_steps_in_period - self.current_step)
        if time_left<=2:
            # if profitable => just bid best ask, else fallback
            if c_ask and val>=c_ask:
                return (c_ask, self)
            else:
                return (c_bid+0.01, self)  # or clamp if c_bid is near val

        # if difference is small => jump in
        if c_ask and (c_ask - c_bid)/(c_ask+0.001) < 0.1 and val>=c_ask:
            return (c_ask, self)

        # default fallback => mid of (c_bid, val)
        mid_price = 0.5*(c_bid + val)
        if mid_price>val:
            mid_price=val
        # clamp between 0..1
        mid_price = min(1.0, max(0.0, mid_price))

        return (mid_price, self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        if val is None:
            return False
        time_left = (self.total_steps_in_period - self.current_step)
        # if profitable or time is short
        if val>=best_ask or time_left<=2:
            return True
        return False

    def decide_to_sell(self, best_bid):
        return False


class KaplanSeller(BaseTrader):
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="kaplan-seller")
        self.min_price = 1.0
        self.max_price = 0.0
        self.sum_prices = 0.0
        self.num_trades = 0

    def update_trade_stats(self, trade_price):
        self.min_price = min(self.min_price, trade_price)
        self.max_price = max(self.max_price, trade_price)
        self.sum_prices += trade_price
        self.num_trades += 1

    def average_price(self):
        return (self.sum_prices / self.num_trades) if self.num_trades>0 else 0.5

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None

        if c_ask is None or c_ask<=0:
            # fallback e.g. cost + half leftover
            fallback = max(cost+0.01, 0.5*(cost + 1.0))
            if fallback>1.0:
                fallback=1.0
            return (fallback, self)

        time_left = (self.total_steps_in_period - self.current_step)
        if time_left<=2:
            # if profitable => ask = best_bid
            if c_bid and c_bid>=cost:
                return (c_bid, self)
            return (c_ask-0.01, self)

        if c_bid and (c_ask - c_bid)/(c_bid+0.001)<0.1 and c_bid>cost:
            return (c_bid, self)

        # fallback => mid
        mid_price = 0.5*(c_ask + cost)
        if mid_price<cost:
            mid_price=cost
        mid_price = min(mid_price, 1.0)
        return (mid_price, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        if cost is None:
            return False
        time_left = (self.total_steps_in_period - self.current_step)
        if best_bid>=cost or time_left<=2:
            return True
        return False

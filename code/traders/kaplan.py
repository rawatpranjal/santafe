# traders/kaplan.py

import random
from .base import BaseTrader

class ImprovedKaplanBuyer(BaseTrader):
    def __init__(self, name, is_buyer, private_values, margin=0.1):
        super().__init__(name, is_buyer, private_values, "kaplan-buyer")
        self.margin_init=margin

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val=self.next_token_value()
        if val is None:
            return None

        fraction = self.current_step / max(1, self.total_steps_in_period - 1)
        margin_dyn = self.margin_init*(1.0 - fraction)

        # if no ask => fallback
        if (c_ask is None) or (c_ask<=0):
            fallback_bid = random.uniform(0.01, val)
            return (fallback_bid, self)

        near_end = (self.current_step>=self.total_steps_in_period-2)
        if near_end and val>=c_ask:
            return (c_ask,self)

        if (val - c_ask)>=margin_dyn:
            return (c_ask,self)
        else:
            fallback = c_bid if c_bid else min_price
            mid_price = 0.5*(fallback+val)
            if mid_price>val:
                mid_price=val
            return (max(0.001,mid_price), self)

    def decide_to_buy(self, best_ask):
        val=self.next_token_value()
        return (val is not None) and (val>=best_ask)

    def decide_to_sell(self, best_bid):
        return False


class ImprovedKaplanSeller(BaseTrader):
    def __init__(self, name, is_buyer, private_values, margin=0.1):
        super().__init__(name, is_buyer, private_values, "kaplan-seller")
        self.margin_init=margin

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost=self.next_token_value()
        if cost is None:
            return None

        fraction = self.current_step / max(1, self.total_steps_in_period - 1)
        margin_dyn = self.margin_init*(1.0 - fraction)

        if (c_bid is None) or (c_bid<=0):
            fallback_ask = random.uniform(cost, max_price)
            return (fallback_ask, self)

        near_end = (self.current_step>=self.total_steps_in_period-2)
        if near_end and c_bid>=cost:
            return (c_bid,self)

        if (c_bid - cost)>=margin_dyn:
            return (c_bid,self)
        else:
            fallback = c_ask if c_ask else max_price
            mid_ask = 0.5*(cost + fallback)
            if mid_ask<cost:
                mid_ask=cost
            return (min(mid_ask, max_price-0.001), self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost=self.next_token_value()
        return (cost is not None) and (best_bid>=cost)

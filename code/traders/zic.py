# traders/zic.py

import random
from .base import BaseTrader

class RandomBuyer(BaseTrader):
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="random-buyer")

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val=self.next_token_value()
        if val is None:
            return None
        # random from [0, val]
        return (random.uniform(0.0, val), self)

    def decide_to_buy(self, best_ask):
        val=self.next_token_value()
        return (val is not None) and (val>=best_ask)

    def decide_to_sell(self, best_bid):
        return False


class RandomSeller(BaseTrader):
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="random-seller")

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost=self.next_token_value()
        if cost is None:
            return None
        # random from [cost, max_price]
        return (random.uniform(cost, max_price), self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost=self.next_token_value()
        return (cost is not None) and (best_bid>=cost)

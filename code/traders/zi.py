import random
from .base import BaseTrader

class ZIBuyer(BaseTrader):
    """
    Zero-Intelligence Buyer:
      - Bids uniformly in [0,1], ignoring any valuation.
      - Decides to buy if it still has tokens left (no budget constraints).
    """
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="zi-buyer")

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        # Uniform in [0,1], ignoring private_values
        return (random.uniform(0.0, 1.0), self)

    def decide_to_buy(self, best_ask):
        # Always buy if you can (no constraints)
        return self.can_trade()

    def decide_to_sell(self, best_bid):
        return False


class ZISeller(BaseTrader):
    """
    Zero-Intelligence Seller:
      - Asks uniformly in [0,1], ignoring any cost.
      - Decides to sell if it still has tokens left (no budget constraints).
    """
    def __init__(self, name, is_buyer, private_values):
        super().__init__(name, is_buyer, private_values, strategy="zi-seller")

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        # Uniform in [0,1], ignoring private_values
        return (random.uniform(0.0, 1.0), self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        # Always sell if you can
        return self.can_trade()

# traders/zip.py

import random
from .base import BaseTrader

class ZipBuyer(BaseTrader):
    """
    A simple ZIP-style buyer:
      - Maintains a margin in [0,1] to shift its bid away from (value).
      - If 'won' last time (bought cheaply), margin goes down (less aggressive).
      - If 'lost' last time (missed a profitable trade), margin goes up (more cautious).
      - The actual bid ~ value*(1 - margin).
    """
    def __init__(self, name, is_buyer, private_values, margin_init=0.05, beta=0.1, gamma=0.1):
        super().__init__(name, is_buyer, private_values, strategy="zip-buyer")
        self.margin = margin_init  # how far below 'value' we bid
        self.beta = beta           # learning rate
        self.gamma = gamma         # momentum / smoothing
        self.momentum = 0.0        # to store momentum from last update
        self.last_trade_outcome = None  # "win"/"lose"/None
        self.last_bid_price = None

    def update_margin(self, outcome, price):
        """
        outcome: "win" if we actually traded, "lose" if not
        price: trade price or best ask we failed to get
        We'll do a simplistic approach:
          - If we 'win', we reduce margin slightly => margin -= delta
          - If we 'lose' (and might have been profitable), we raise margin => margin += delta
        """
        # We'll define a 'desired margin' for this step, d_j, by guessing
        # that an "optimal" price was half or so. This is just a placeholder.
        val = self.next_token_value()
        if val is None:
            return
        # pretend the "optimal" price is half the difference
        # e.g. best guess ~ mid of price & val
        # Then desired margin: d_j = 1 - (optimalBid / val).
        # For simplicity, let's assume:
        optimal_bid = 0.5*(price + val) if price<val else val
        desired_margin = 1.0 - (optimal_bid / val)

        # delta = beta*(d_j - current_margin)
        raw_delta = self.beta * (desired_margin - self.margin)
        # apply momentum
        delta = self.gamma*self.momentum + (1.0 - self.gamma)*raw_delta
        self.momentum = delta  # store for next time

        # If outcome == "win", interpret as we got it for cheaper => margin smaller
        if outcome=="win":
            self.margin += (delta * -1.0)
        elif outcome=="lose":
            self.margin += (delta * +1.0)

        # clamp margin in [0,1]
        self.margin = max(0.0, min(1.0, self.margin))

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None
        # Our current bid price
        bid_price = val * (1.0 - self.margin)
        # store for reference
        self.last_bid_price = bid_price
        return (max(bid_price, 0.0), self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        # if we can profit => decide to buy
        if val is not None and val>=best_ask:
            return True
        return False

    def decide_to_sell(self, best_bid):
        return False

    def transact(self, price):
        """Override so we can detect 'win'/'lose' easily."""
        # If the trade actually happens => outcome='win'
        outcome = super().transact(price)
        if outcome>0.0:
            # we made a purchase
            self.last_trade_outcome = "win"
            self.update_margin("win", price)
        return outcome


class ZipSeller(BaseTrader):
    """
    A simple ZIP-style seller:
      - Maintains a margin in [0,1] above the cost => ask = cost*(1 + margin).
      - If 'win' last time (sold for a good price), margin might decrease
        (be less aggressive next time).
      - If 'lose' (missed a profitable sale), margin goes up (be more cautious).
    """
    def __init__(self, name, is_buyer, private_values, margin_init=0.05, beta=0.1, gamma=0.1):
        super().__init__(name, is_buyer, private_values, strategy="zip-seller")
        self.margin = margin_init
        self.beta = beta
        self.gamma = gamma
        self.momentum = 0.0
        self.last_trade_outcome = None
        self.last_ask_price = None

    def update_margin(self, outcome, price):
        cost = self.next_token_value()
        if cost is None:
            return
        # naive guess at "optimal" => let's say mid of price & cost
        #   if price>cost
        if price<cost:
            optimal_ask = cost
        else:
            optimal_ask = 0.5*(cost + price)

        desired_margin = (optimal_ask / cost) - 1.0  # i.e.  ask= cost*(1+margin)

        raw_delta = self.beta * (desired_margin - self.margin)
        delta = self.gamma*self.momentum + (1.0 - self.gamma)*raw_delta
        self.momentum = delta

        if outcome=="win":
            # sold => reduce margin
            self.margin += (delta * -1.0)
        elif outcome=="lose":
            self.margin += (delta * +1.0)

        self.margin = max(0.0, min(1.0, self.margin))

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None
        ask_price = cost * (1.0 + self.margin)
        self.last_ask_price = ask_price
        # clamp
        if ask_price>max_price:
            ask_price = max_price
        return (ask_price, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid>=cost)

    def transact(self, price):
        outcome = super().transact(price)
        if outcome>0.0:
            self.last_trade_outcome = "win"
            self.update_margin("win", price)
        return outcome

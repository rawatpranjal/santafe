# traders.py
import random

class BaseTrader:
    def __init__(self, name, is_buyer, private_values):
        self.name = name
        self.is_buyer = is_buyer
        self.private_values = private_values
        self.max_tokens = len(private_values)
        self.tokens_left = len(private_values)
        self.profit = 0.0

        self.current_round = 0
        self.current_period = 0
        self.current_step = 0
        self.total_steps_in_period = 0

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        self.tokens_left = self.max_tokens
        self.total_steps_in_period = total_steps_in_period
        self.current_round = round_idx
        self.current_period = period_idx
        self.current_step = 0

    def can_trade(self):
        return self.tokens_left > 0

    def next_token_value(self):
        idx = self.max_tokens - self.tokens_left
        return self.private_values[idx] if 0 <= idx < len(self.private_values) else None

    def transact(self, price):
        if not self.can_trade():
            return 0.0
        val = self.next_token_value()
        if val is None:
            return 0.0

        inc_profit = (val - price) if self.is_buyer else (price - val)
        self.profit += inc_profit
        self.tokens_left -= 1
        return inc_profit

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        raise NotImplementedError

    def decide_to_buy(self, best_ask):
        raise NotImplementedError

    def decide_to_sell(self, best_bid):
        raise NotImplementedError


class ZeroIntelligenceTrader(BaseTrader):
    """
    Random quotes in the feasible range.
    """
    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None

        if self.is_buyer:
            # random bid in [0, val]
            return (random.uniform(0, val), self)
        else:
            # random ask in [val, 1]
            return (random.uniform(val, 1), self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        return (val is not None) and (val >= best_ask)

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid >= cost)


class TruthTellerBuyer(BaseTrader):
    """
    Always bids exactly its value (if it can trade).
    """
    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None
        return (val, self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        return (val is not None) and (val >= best_ask)

    def decide_to_sell(self, best_bid):
        return False


class TruthTellerSeller(BaseTrader):
    """
    Always asks exactly its cost.
    """
    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None
        return (cost, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid >= cost)


class KaplanBuyer(BaseTrader):
    """
    Snipe if (value - c_ask) >= margin or near end.
    Else place tiny decoy.
    """
    def __init__(self, name, is_buyer, private_values, margin=0.05):
        super().__init__(name, is_buyer, private_values)
        self.margin = margin

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None
        if (c_ask is None) or (c_ask <= 0):
            return None

        near_end = (self.current_step >= self.total_steps_in_period - 2)
        if (val - c_ask >= self.margin) or near_end:
            return (c_ask, self)
        else:
            return (0.01, self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        return (val is not None) and (val >= best_ask)

    def decide_to_sell(self, best_bid):
        return False


class KaplanSeller(BaseTrader):
    """
    Snipe if (c_bid - cost) >= margin or near end.
    Else place large decoy.
    """
    def __init__(self, name, is_buyer, private_values, margin=0.05):
        super().__init__(name, is_buyer, private_values)
        self.margin = margin

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None
        if (c_bid is None) or (c_bid <= 0):
            return None

        near_end = (self.current_step >= self.total_steps_in_period - 2)
        if (c_bid - cost >= self.margin) or near_end:
            return (c_bid, self)
        else:
            return (0.99, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid >= cost)


class CreeperBuyer(BaseTrader):
    """
    Gradually increases bid from min_price to value.
    """
    def __init__(self, name, is_buyer, private_values, speed=0.1):
        super().__init__(name, is_buyer, private_values)
        self.speed = speed

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        val = self.next_token_value()
        if val is None:
            return None

        fraction = min(1.0, self.current_step * self.speed)
        price = min_price + fraction * (val - min_price)
        return (price, self)

    def decide_to_buy(self, best_ask):
        val = self.next_token_value()
        return (val is not None) and (val >= best_ask)

    def decide_to_sell(self, best_bid):
        return False


class CreeperSeller(BaseTrader):
    """
    Gradually decreases ask from max_price down to cost.
    """
    def __init__(self, name, is_buyer, private_values, speed=0.1):
        super().__init__(name, is_buyer, private_values)
        self.speed = speed

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        if not self.can_trade():
            return None
        cost = self.next_token_value()
        if cost is None:
            return None

        fraction = min(1.0, self.current_step * self.speed)
        price = max_price - fraction * (max_price - cost)
        return (price, self)

    def decide_to_buy(self, best_ask):
        return False

    def decide_to_sell(self, best_bid):
        cost = self.next_token_value()
        return (cost is not None) and (best_bid >= cost)

# traders/base.py

class BaseTrader:
    def __init__(self, name, is_buyer, private_values, strategy="base"):
        self.name = name
        self.strategy = strategy
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
        self.profit = 0.0  # keep profit across entire round or reset if needed

    def can_trade(self):
        return self.tokens_left>0

    def next_token_value(self):
        idx = self.max_tokens - self.tokens_left
        if idx<0 or idx>=len(self.private_values):
            return None
        return self.private_values[idx]

    def transact(self, price):
        """If buyer => profit=val-price; if seller => profit=price-val."""
        if not self.can_trade():
            return 0.0
        val=self.next_token_value()
        if val is None:
            return 0.0
        inc = (val-price) if self.is_buyer else (price-val)
        self.profit+=inc
        self.tokens_left-=1
        return inc

    def make_bid_or_ask(self, c_bid, c_ask, pmin, pmax, min_price, max_price):
        """Return (price, self) or None."""
        raise NotImplementedError

    def decide_to_buy(self, best_ask):
        raise NotImplementedError

    def decide_to_sell(self, best_bid):
        raise NotImplementedError

# traders/mu.py
import logging
import numpy as np
from .base import BaseTrader

class MUBuyer(BaseTrader):
    """ Markup Buyer: Bids value * (1 - rate), accepts if profitable vs value. """
    def __init__(self, name, is_buyer, private_values, markup_rate=0.1, **kwargs):
        super().__init__(name, True, private_values, strategy="mu")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if not (0 <= markup_rate < 1):
             self.logger.warning(f"Markup rate {markup_rate} invalid. Clamping to [0, 1). Using 0.1 if needed.")
             markup_rate = np.clip(markup_rate, 0.0, 0.999) # Ensure rate < 1
        self.markup_rate = markup_rate
        self.logger.debug(f"Initialized MU Buyer with rate={self.markup_rate:.3f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Bid value * (1 - markup_rate). """
        if not self.can_trade(): return None
        value = self.get_next_value_cost()
        if value is None: return None

        target_bid = value * (1.0 - self.markup_rate)
        # Clamp to market bounds and ensure integer price
        bid_price = max(self.min_price, min(self.max_price, int(round(target_bid))))
        # Final check: ensure bid is profitable (<= value)
        bid_price = min(bid_price, value)
        bid_price = max(self.min_price, bid_price) # Ensure >= min_price

        self.logger.debug(f"MU proposing bid {bid_price} (Value={value}, Rate={self.markup_rate:.3f})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer <= own value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        if is_profitable:
            self.logger.debug(f"MU accepting BUY at {current_offer_price} (Value={value})")
            self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

class MUSeller(BaseTrader):
    """ Markup Seller: Asks cost * (1 + rate), accepts if profitable vs cost. """
    def __init__(self, name, is_buyer, private_values, markup_rate=0.1, **kwargs):
        super().__init__(name, False, private_values, strategy="mu")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if markup_rate < 0:
            self.logger.warning(f"Markup rate {markup_rate} invalid. Clamping to >= 0.")
            markup_rate = max(0.0, markup_rate)
        self.markup_rate = markup_rate
        self.logger.debug(f"Initialized MU Seller with rate={self.markup_rate:.3f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Ask cost * (1 + markup_rate). """
        if not self.can_trade(): return None
        cost = self.get_next_value_cost()
        if cost is None: return None

        target_ask = cost * (1.0 + self.markup_rate)
        ask_price = max(self.min_price, min(self.max_price, int(round(target_ask))))
        # Final check: ensure ask is profitable (>= cost)
        ask_price = max(ask_price, cost)
        ask_price = min(self.max_price, ask_price) # Ensure <= max_price

        self.logger.debug(f"MU proposing ask {ask_price} (Cost={cost}, Rate={self.markup_rate:.3f})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid >= own cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        if is_profitable:
            self.logger.debug(f"MU accepting SELL at {current_bid_price} (Cost={cost})")
            self._clear_rl_step_state()
        return is_profitable
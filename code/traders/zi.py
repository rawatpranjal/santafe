import random
import logging
from .base import BaseTrader

class ZIBuyer(BaseTrader):
    """ZI Buyer Strategy: Random bid/ask, always accepts."""
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="zi")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Submit a random bid between min_price and max_price."""
        if not self.can_trade():
            return None
        try:
            bid_price = random.randint(self.min_price, self.max_price)
            self.logger.debug(f"Proposing random bid {bid_price}")
            return bid_price
        except ValueError:
            self.logger.warning(f"Value error generating random bid [{self.min_price},{self.max_price}]")
            return None # Should only happen if min_price > max_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Always accepts."""
        if not self.can_trade(): return False
        self.logger.debug(f"ZI auto-accepting BUY at {current_offer_price}")
        # Clear any potential RL state from previous steps if auto-accepting
        self._current_step_state = None; self._current_step_action = None;
        self._current_step_log_prob = None; self._current_step_value = None;
        return True

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Buyers do not respond to sell requests."""
        return False


class ZISeller(BaseTrader):
    """ZI Seller Strategy: Random bid/ask, always accepts."""
    def __init__(self, name, is_buyer, private_values, **kwargs):
        # Note: is_buyer argument is False for sellers
        super().__init__(name, False, private_values, strategy="zi")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Submit a random ask between min_price and max_price."""
        if not self.can_trade():
            return None
        try:
            ask_price = random.randint(self.min_price, self.max_price)
            self.logger.debug(f"Proposing random ask {ask_price}")
            return ask_price
        except ValueError:
            self.logger.warning(f"Value error generating random ask [{self.min_price},{self.max_price}]")
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Sellers do not respond to buy requests."""
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Always accepts."""
        if not self.can_trade(): return False
        self.logger.debug(f"ZI auto-accepting SELL at {current_bid_price}")
        # Clear any potential RL state from previous steps if auto-accepting
        self._current_step_state = None; self._current_step_action = None;
        self._current_step_log_prob = None; self._current_step_value = None;
        return True

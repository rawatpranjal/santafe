# traders/tt.py
import logging
from .base import BaseTrader

class TTBuyer(BaseTrader):
    """ Truth Teller Buyer: Bids own value, accepts if profitable. """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="tt")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Always bid own value for the next token. """
        if not self.can_trade(): return None
        value = self.get_next_value_cost()
        if value is None: return None

        # Bid exactly the value, clamped to market bounds
        bid_price = max(self.min_price, min(self.max_price, value))
        self.logger.debug(f"TT proposing bid {bid_price} (Value={value})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer <= own value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        if is_profitable:
            self.logger.debug(f"TT requesting BUY at {current_offer_price} (Value={value})")
            self._clear_rl_step_state() # Clear potential inherited RL state
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Buyers don't accept sell requests

class TTSeller(BaseTrader):
    """ Truth Teller Seller: Asks own cost, accepts if profitable. """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="tt")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Always ask own cost for the next token. """
        if not self.can_trade(): return None
        cost = self.get_next_value_cost()
        if cost is None: return None

        # Ask exactly the cost, clamped to market bounds
        ask_price = max(self.min_price, min(self.max_price, cost))
        self.logger.debug(f"TT proposing ask {ask_price} (Cost={cost})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Sellers don't accept buy requests

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid >= own cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        if is_profitable:
            self.logger.debug(f"TT requesting SELL at {current_bid_price} (Cost={cost})")
            self._clear_rl_step_state() # Clear potential inherited RL state
        return is_profitable
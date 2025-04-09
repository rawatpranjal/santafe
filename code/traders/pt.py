# traders/pt.py
import logging
from .base import BaseTrader

# Simple Pricetaker: Only accepts offers/bids if they are profitable
# against the agent's own reservation price. Never submits quotes.

class PTBuyer(BaseTrader):
    """ Pricetaker Buyer: Accepts profitable asks, never quotes. """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="pt")
        self.logger = logging.getLogger(f'trader.{self.name}')
        self.logger.debug("Initialized PT Buyer.")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Pricetakers do not submit quotes in this simple version. """
        # self.logger.debug("PT: Not submitting quote.")
        return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer <= own value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        if value is None: return False

        is_profitable = False
        try:
            is_profitable = (float(current_offer_price) <= value)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid offer price {current_offer_price} in request_buy.")
            return False

        if is_profitable:
            self.logger.debug(f"PT requesting BUY at {current_offer_price} (Value={value})")
            self._clear_rl_step_state() # Clear potential inherited RL state
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Buyers don't accept sell requests

class PTSeller(BaseTrader):
    """ Pricetaker Seller: Accepts profitable bids, never quotes. """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="pt")
        self.logger = logging.getLogger(f'trader.{self.name}')
        self.logger.debug("Initialized PT Seller.")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Pricetakers do not submit quotes in this simple version. """
        # self.logger.debug("PT: Not submitting quote.")
        return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Sellers don't accept buy requests

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid >= own cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        if cost is None: return False

        is_profitable = False
        try:
            is_profitable = (float(current_bid_price) >= cost)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid bid price {current_bid_price} in request_sell.")
            return False

        if is_profitable:
            self.logger.debug(f"PT requesting SELL at {current_bid_price} (Cost={cost})")
            self._clear_rl_step_state() # Clear potential inherited RL state
        return is_profitable
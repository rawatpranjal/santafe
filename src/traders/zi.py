# traders/zi.py
import logging
import random
import numpy as np
from .base import BaseTrader

class ZIUBuyer(BaseTrader):
    """
    Zero Intelligence Unconstrained (ZI-U) Buyer.
    Submits random bids between min_price and max_price, ignoring value.
    Accepts any offered trade.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="zi") # is_buyer=True
        self.logger = logging.getLogger(f'trader.{self.name}')
        # self.rng = random.Random(self.id_numeric)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Generates a random bid between min_price and max_price. """
        if not self.is_buyer: return None
        if not self.can_trade(): return None

        # Random quote across the entire valid range
        # Use uniform for potentially better distribution than int(random*range)
        bid = random.uniform(self.min_price, self.max_price)
        final_bid = int(round(bid))

        # Clamp just in case uniform goes slightly out bounds due to float precision
        final_bid = max(self.min_price, min(self.max_price, final_bid))

        # self.logger.debug(f"ZI-U Buyer proposing bid: {final_bid}")
        return final_bid

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Always accept an ask if holding the current bid. """
        accept = self.can_trade() # Only constraint is having tokens left
        # self.logger.debug(f"ZI-U Buyer request_buy(Offer={current_offer_price}): -> Accept={accept}")
        return accept

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ ZI-U Buyers do not sell. """
        return False


class ZIUSeller(BaseTrader):
    """
    Zero Intelligence Unconstrained (ZI-U) Seller.
    Submits random asks between min_price and max_price, ignoring cost.
    Accepts any offered trade.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="zi") # is_buyer=False
        self.logger = logging.getLogger(f'trader.{self.name}')
        # self.rng = random.Random(self.id_numeric)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Generates a random ask between min_price and max_price. """
        if self.is_buyer: return None
        if not self.can_trade(): return None

        # Random quote across the entire valid range
        ask = random.uniform(self.min_price, self.max_price)
        final_ask = int(round(ask))

        # Clamp just in case
        final_ask = max(self.min_price, min(self.max_price, final_ask))

        # self.logger.debug(f"ZI-U Seller proposing ask: {final_ask}")
        return final_ask

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ ZI-U Sellers do not buy. """
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Always accept a bid if holding the current ask. """
        accept = self.can_trade() # Only constraint is having tokens left
        # self.logger.debug(f"ZI-U Seller request_sell(Bid={current_bid_price}): -> Accept={accept}")
        return accept
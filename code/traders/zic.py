# traders/zic.py
import logging
import random
import numpy as np
from .base import BaseTrader

class ZICBuyer(BaseTrader):
    """
    Zero Intelligence Constrained (ZIC) Buyer.
    Based on Gode & Sunder (1993) and SRobotZI1 logic.
    Submits random bids between min_price and its value.
    Accepts trades only if profitable.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="zic") # is_buyer=True
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Use a dedicated RNG for this agent if needed, seeded uniquely
        # self.rng = random.Random(self.id_numeric) # Example seeding
        # Or just use global random

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Generates a random bid between min_price and agent's value. """
        if not self.is_buyer: return None # Should not happen if registry is correct
        if not self.can_trade(): return None

        value = self.get_next_value_cost()
        if value is None: return None # Cannot determine value

        # Mimic Java: value - random * (value - min_price)
        # Ensure float division and handle value == min_price case
        value_f = float(value)
        min_price_f = float(self.min_price)
        if value_f <= min_price_f:
            # If value is at or below min price, can only bid min_price profitably
            newbid = min_price_f
        else:
            # random.random() gives [0.0, 1.0)
            newbid = value_f - random.random() * (value_f - min_price_f)

        # Convert to int and clamp (Java clamps at the end)
        bid_int = int(round(newbid))
        final_bid = max(self.min_price, min(value, bid_int)) # Ensure doesn't exceed value or go below min

        # self.logger.debug(f"ZIC Buyer proposing bid: {final_bid} (Value={value})")
        return final_bid

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept ask only if it's profitable. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        if value is None: return False

        try: offer_price_f = float(current_offer_price)
        except (ValueError, TypeError): return False

        accept = (offer_price_f <= value) # Profitability check
        # self.logger.debug(f"ZIC Buyer request_buy(Offer={offer_price_f}): Value={value} -> Accept={accept}")
        return accept

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ ZIC Buyers do not sell. """
        return False


class ZICSeller(BaseTrader):
    """
    Zero Intelligence Constrained (ZIC) Seller.
    Based on Gode & Sunder (1993) and SRobotZI1 logic.
    Submits random asks between its cost and max_price.
    Accepts trades only if profitable.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="zic") # is_buyer=False
        self.logger = logging.getLogger(f'trader.{self.name}')
        # self.rng = random.Random(self.id_numeric)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Generates a random ask between agent's cost and max_price. """
        if self.is_buyer: return None
        if not self.can_trade(): return None

        cost = self.get_next_value_cost()
        if cost is None: return None

        # Mimic Java: cost + random * (max_price - cost)
        cost_f = float(cost)
        max_price_f = float(self.max_price)
        if cost_f >= max_price_f:
            # If cost is at or above max price, can only ask max_price profitably
            newask = max_price_f
        else:
            newask = cost_f + random.random() * (max_price_f - cost_f)

        # Convert to int and clamp
        ask_int = int(round(newask))
        final_ask = min(self.max_price, max(cost, ask_int)) # Ensure doesn't go below cost or above max

        # self.logger.debug(f"ZIC Seller proposing ask: {final_ask} (Cost={cost})")
        return final_ask

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ ZIC Sellers do not buy. """
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept bid only if it's profitable. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        if cost is None: return False

        try: bid_price_f = float(current_bid_price)
        except (ValueError, TypeError): return False

        accept = (bid_price_f >= cost) # Profitability check
        # self.logger.debug(f"ZIC Seller request_sell(Bid={bid_price_f}): Cost={cost} -> Accept={accept}")
        return accept
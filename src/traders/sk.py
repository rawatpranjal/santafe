# traders/sk.py
import logging
import numpy as np
import random # Needed if adding shout probability later
from .base import BaseTrader

# ASSUMPTION: Skeleton behaves like a non-adaptive ZIP trader.
# Quotes based on a *fixed* profit margin.
# Accepts profitable trades like ZIC/Pricetaker.

class SKBuyer(BaseTrader):
    """
    Skeleton Buyer Strategy (ASSUMED to be non-adaptive ZIP).
    Quotes based on fixed margin, accepts profitable trades.
    """
    def __init__(self, name, is_buyer, private_values, fixed_margin=0.05, shout_probability=0.9, **kwargs):
        # Note: Chen & Tai description is vague. This assumes non-adaptive margin quoting.
        super().__init__(name, True, private_values, strategy="sk")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if not (0 <= fixed_margin < 1):
            self.logger.warning(f"Skeleton fixed_margin {fixed_margin} invalid. Clamping to [0, 1). Using 0.05.")
            fixed_margin = np.clip(fixed_margin, 0.0, 0.999)
        self.margin = fixed_margin
        self.shout_probability = np.clip(shout_probability, 0.0, 1.0)
        self.logger.debug(f"Initialized SK Buyer with fixed_margin={self.margin:.3f}, shout_prob={self.shout_probability:.2f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Calculate and potentially submit a bid based on fixed margin. """
        if not self.can_trade(): return None

        # Decide whether to shout based on probability
        if random.random() >= self.shout_probability:
            # self.logger.debug("SK decided not to shout bid.")
            return None

        value = self.get_next_value_cost()
        if value is None: return None

        # Calculate bid based on fixed margin
        target_bid = value * (1.0 - self.margin)
        bid_price = max(self.min_price, min(self.max_price, int(round(target_bid))))
        bid_price = min(bid_price, value) # Ensure profitable
        bid_price = max(self.min_price, bid_price)

        self.logger.debug(f"SK proposing bid {bid_price} (Value={value}, Margin={self.margin:.3f})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer <= own value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        if value is None: return False

        is_profitable = False
        try: is_profitable = (float(current_offer_price) <= value)
        except (ValueError, TypeError): return False

        if is_profitable:
            self.logger.debug(f"SK accepting BUY at {current_offer_price} (Value={value})")
            self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

class SKSeller(BaseTrader):
    """
    Skeleton Seller Strategy (ASSUMED to be non-adaptive ZIP).
    Quotes based on fixed margin, accepts profitable trades.
    """
    def __init__(self, name, is_buyer, private_values, fixed_margin=0.05, shout_probability=0.9, **kwargs):
        super().__init__(name, False, private_values, strategy="sk")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if fixed_margin < 0:
            self.logger.warning(f"Skeleton fixed_margin {fixed_margin} invalid. Clamping to >= 0. Using 0.05.")
            fixed_margin = max(0.0, fixed_margin)
        self.margin = fixed_margin
        self.shout_probability = np.clip(shout_probability, 0.0, 1.0)
        self.logger.debug(f"Initialized SK Seller with fixed_margin={self.margin:.3f}, shout_prob={self.shout_probability:.2f}")

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Calculate and potentially submit an ask based on fixed margin. """
        if not self.can_trade(): return None

        if random.random() >= self.shout_probability:
            # self.logger.debug("SK decided not to shout ask.")
            return None

        cost = self.get_next_value_cost()
        if cost is None: return None

        # Calculate ask price
        target_ask = cost * (1.0 + self.margin)
        ask_price = max(self.min_price, min(self.max_price, int(round(target_ask))))
        ask_price = max(ask_price, cost) # Ensure profitable
        ask_price = min(self.max_price, ask_price)

        self.logger.debug(f"SK proposing ask {ask_price} (Cost={cost}, Margin={self.margin:.3f})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
         return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid >= own cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        if cost is None: return False

        is_profitable = False
        try: is_profitable = (float(current_bid_price) >= cost)
        except (ValueError, TypeError): return False

        if is_profitable:
            self.logger.debug(f"SK accepting SELL at {current_bid_price} (Cost={cost})")
            self._clear_rl_step_state()
        return is_profitable
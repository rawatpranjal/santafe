# traders/zic.py
import random
import logging
from .base import BaseTrader

class ZICBuyer(BaseTrader):
    """ZIC Buyer Strategy: Random profitable bid/ask, accepts profitable trades."""
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="zic")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Probability of submitting a quote in the bid/offer substep
        self.shout_probability = kwargs.get('shout_probability', 0.8)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Submit a random bid between min_price and own value."""
        if not self.can_trade():
            return None

        value = self.get_next_value_cost()
        if value is None:
            self.logger.warning("Can trade but get_next_value_cost returned None.")
            return None

        min_possible_bid = self.min_price
        max_possible_bid = value

        if max_possible_bid < min_possible_bid:
            # Cannot bid profitably
            self.logger.debug(f"Cannot bid profitably (V {value} < MinP {min_possible_bid})")
            return None

        if random.random() < self.shout_probability:
            try:
                # Ensure range is valid before calling randint
                if min_possible_bid <= max_possible_bid:
                    bid_price = random.randint(min_possible_bid, max_possible_bid)
                    self.logger.debug(f"Proposing bid {bid_price} (Range: [{min_possible_bid}, {max_possible_bid}])")
                    return bid_price
                else:
                    # This case should be caught by the check above, but added defensively
                    self.logger.warning(f"Invalid range for randint bid [{min_possible_bid},{max_possible_bid}]")
                    return None
            except ValueError:
                # Handle potential edge case errors in randint
                self.logger.warning(f"Value error generating bid for range [{min_possible_bid},{max_possible_bid}]")
                return None
        else:
            # Did not meet shout probability
            self.logger.debug("Decided not to shout bid.")
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Accept any offer that is less than or equal to own value."""
        if not self.can_trade() or current_offer_price is None:
            return False

        value = self.get_next_value_cost()
        if value is None:
            return False # Should not happen if can_trade is True

        is_profitable = (current_offer_price <= value)
        if is_profitable:
            self.logger.debug(f"Requesting BUY at {current_offer_price} (Value={value})")
        else:
            self.logger.debug(f"Rejecting BUY at {current_offer_price} (Value={value})")
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Buyers do not respond to sell requests."""
        return False


class ZICSeller(BaseTrader):
    """ZIC Seller Strategy: Random profitable ask, accepts profitable bids."""
    def __init__(self, name, is_buyer, private_values, **kwargs):
        # Note: is_buyer argument is False for sellers
        super().__init__(name, False, private_values, strategy="zic")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Probability of submitting a quote in the bid/offer substep
        self.shout_probability = kwargs.get('shout_probability', 0.8)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Submit a random ask between own cost and max_price."""
        if not self.can_trade():
            return None

        cost = self.get_next_value_cost()
        if cost is None:
            self.logger.warning("Can trade but get_next_value_cost returned None.")
            return None

        min_possible_ask = cost
        max_possible_ask = self.max_price

        if min_possible_ask > max_possible_ask:
            # Cannot ask profitably (should not happen unless cost > max_price)
            self.logger.debug(f"Cannot ask profitably (C {cost} > MaxP {max_possible_ask})")
            return None

        if random.random() < self.shout_probability:
            try:
                 # Ensure range is valid before calling randint
                if min_possible_ask <= max_possible_ask:
                    ask_price = random.randint(min_possible_ask, max_possible_ask)
                    self.logger.debug(f"Proposing ask {ask_price} (Range: [{min_possible_ask}, {max_possible_ask}])")
                    return ask_price
                else:
                    self.logger.warning(f"Invalid range for randint ask [{min_possible_ask},{max_possible_ask}]")
                    return None
            except ValueError:
                self.logger.warning(f"Value error generating ask for range [{min_possible_ask},{max_possible_ask}]")
                return None
        else:
            # Did not meet shout probability
            self.logger.debug("Decided not to shout ask.")
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Sellers do not respond to buy requests."""
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Accept any bid that is greater than or equal to own cost."""
        if not self.can_trade() or current_bid_price is None:
            return False

        cost = self.get_next_value_cost()
        if cost is None:
            return False # Should not happen if can_trade is True

        is_profitable = (current_bid_price >= cost)
        if is_profitable:
            self.logger.debug(f"Requesting SELL at {current_bid_price} (Cost={cost})")
        else:
            self.logger.debug(f"Rejecting SELL at {current_bid_price} (Cost={cost})")
        return is_profitable
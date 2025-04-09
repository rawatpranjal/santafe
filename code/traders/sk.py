# traders/sk.py
import logging
from .base import BaseTrader

class SKBuyer(BaseTrader):
    """
    Skeleton Buyer Strategy (Interpreted from Chen & Tai description):
    Bids/asks by referring to own reservation prices and current market quotes.
    Interpretation: Improve market bid using own value OR current ask.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="sk")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Bid based on own value and current ask, only if improving current bid. """
        if not self.can_trade(): return None
        value = self.get_next_value_cost()
        if value is None: return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        # Determine target bid: consider own value and current ask
        # Be willing to bid up to own value
        target_bid = value
        # If an ask exists below own value, consider bidding slightly above it (or matching it?)
        # Let's try a simple improvement logic: bid slightly above current bid if possible,
        # but capped by value and potentially influenced by the ask.
        if current_ask is not None:
            # A plausible Skeleton might bid slightly more aggressively if ask is low
            target_bid = min(value, current_ask + 1) # Example: willingness to match or slightly beat ask if profitable

        # Must improve current best bid
        if current_bid is not None and target_bid <= current_bid:
             self.logger.debug(f"SK: Target bid {target_bid} doesn't improve current bid {current_bid}. Waiting.")
             return None

        # Clamp to market bounds and ensure profitable
        bid_price = max(self.min_price, min(self.max_price, int(round(target_bid))))
        bid_price = min(bid_price, value)  # Ensure profitable vs own value
        bid_price = max(self.min_price, bid_price)

        # Re-check improvement after clamping
        if current_bid is not None and bid_price <= current_bid:
             self.logger.debug(f"SK: Clamped bid {bid_price} doesn't improve current bid {current_bid}. Waiting.")
             return None

        self.logger.debug(f"SK proposing bid {bid_price} (Value={value}, CBid={current_bid}, CAsk={current_ask})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer <= own value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

class SKSeller(BaseTrader):
    """
    Skeleton Seller Strategy (Interpreted):
    Asks by referring to own cost and current bid, only if improving current ask.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="sk")
        self.logger = logging.getLogger(f'trader.{self.name}')

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Ask based on own cost and current bid, only if improving current ask. """
        if not self.can_trade(): return None
        cost = self.get_next_value_cost()
        if cost is None: return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        # Determine target ask: consider own cost and current bid
        target_ask = cost
        if current_bid is not None:
             target_ask = max(cost, current_bid - 1) # Example: undercut bid if possible

        # Must improve current best ask
        if current_ask is not None and target_ask >= current_ask:
             self.logger.debug(f"SK: Target ask {target_ask} doesn't improve current ask {current_ask}. Waiting.")
             return None

        # Clamp to market bounds and ensure profitable
        ask_price = max(self.min_price, min(self.max_price, int(round(target_ask))))
        ask_price = max(ask_price, cost) # Ensure profitable vs own cost
        ask_price = min(self.max_price, ask_price)

        # Re-check improvement after clamping
        if current_ask is not None and ask_price >= current_ask:
             self.logger.debug(f"SK: Clamped ask {ask_price} doesn't improve current ask {current_ask}. Waiting.")
             return None

        self.logger.debug(f"SK proposing ask {ask_price} (Cost={cost}, CBid={current_bid}, CAsk={current_ask})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid >= own cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        if is_profitable: self._clear_rl_step_state()
        return is_profitable
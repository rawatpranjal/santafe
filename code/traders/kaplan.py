# traders/kaplan.py
import random
import logging
from .base import BaseTrader

class KaplanBuyer(BaseTrader):
    """
    Kaplan Buyer Strategy: Tries to 'snipe' asks or improve bids slightly.
    Based on RMP94 description: waits, then jumps in when bid/ask are close.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="kaplan")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Parameters for margin check when sniping (optional, defaults are placeholders)
        # These % might need tuning or a different logic based on RMP94 deeper analysis
        self.profit_margin_pct_most = kwargs.get('profit_margin_pct_most', 0.01) # e.g., 1% of own value
        self.profit_margin_pct_quote = kwargs.get('profit_margin_pct_quote', 0.01) # e.g., 1% of quote price

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Kaplan Buyer logic:
        1. If profitable to buy at current ask+1 (snipe), submit bid = ask+1.
        2. Else if no ask, improve current bid by 1 if possible.
        3. Else, do nothing (wait).
        """
        if not self.can_trade(): return None

        limit_price = self.get_next_value_cost() # Buyer's maximum willingness to pay
        if limit_price is None: self.logger.warning("Can trade but get_next_value_cost returned None."); return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        target_bid = None

        # Condition 1: Snipe the Ask?
        # Check if there's an ask below our limit price
        if current_ask is not None and limit_price >= current_ask:
            # Potential profit exists. Determine bid price.
            # Kaplan aims to "steal the deal" - often interpreted as matching or slightly improving
            # Let's try bidding just above the current ask, capped by own limit price
            potential_bid = current_ask # Match ask first - RMP94 says "bid equal to previous ask"
                                        # but paper uses discrete time, maybe bid=ask works better?
                                        # Let's try bidding AT the ask price (if profitable)
                                        # No, the paper says "bidding an amount GREATER than or equal to the previous current ask"
                                        # If we bid AT ask, seller might not accept.
                                        # Let's try bidding exactly the current ask price, hoping seller accepts that bid later
            # Edit: Simplest interpretation of "jump in and steal the deal": match the ask if profitable
            # But RMP94 actually describes the implementation: "places a bid equal to the smaller of the current ask or its current token value"
            # Let's follow that more closely.

            target_bid = min(limit_price, current_ask)
            # Safety check: Ensure target_bid > current_bid if one exists
            if current_bid is not None and target_bid <= current_bid:
                 self.logger.debug(f"Potential snipe bid {target_bid} is not better than current bid {current_bid}. Waiting.")
                 target_bid = None # Don't place non-improving bid
            else:
                 self.logger.debug(f"Targeting snipe bid {target_bid} (based on ASK {current_ask}, LIMIT {limit_price})")

        # Condition 2: Improve Current Bid? (Only if not sniping)
        elif target_bid is None and current_bid is not None:
             # Try to improve the current bid by the smallest increment (e.g., 1)
             if limit_price > current_bid: # Can we bid higher?
                 potential_bid = current_bid + 1
                 target_bid = min(limit_price, potential_bid) # Bid improvement, capped by limit
                 self.logger.debug(f"Targeting improved bid {target_bid} (improving CBID {current_bid}, LIMIT {limit_price})")
             else:
                  self.logger.debug(f"Cannot improve current bid {current_bid} (LIMIT {limit_price})")

        # Condition 3: Initial Bid? (Only if not sniping and no current bid)
        elif target_bid is None and current_bid is None:
             # RMP94 Kaplan doesn't seem to specify initial bid logic when market is empty.
             # It primarily reacts. Let's default to 'pass' in this case.
             self.logger.debug("No ask to snipe and no bid to improve. Waiting.")
             target_bid = None


        # Final submission
        if target_bid is not None:
            # Ensure bid is within market bounds
            final_bid = max(self.min_price, min(self.max_price, int(round(target_bid))))
            self.logger.debug(f"Proposing final bid {final_bid}")
            return final_bid
        else:
            # Do not submit a bid this step
            self.logger.debug("Not submitting bid.")
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Kaplan accepts any profitable offer."""
        if not self.can_trade() or current_offer_price is None:
            return False
        value = self.get_next_value_cost()
        if value is None:
            return False
        is_profitable = (current_offer_price <= value)
        if is_profitable:
            self.logger.debug(f"Requesting BUY at {current_offer_price} (Value={value})")
        # else: self.logger.debug(f"Rejecting BUY at {current_offer_price} (Value={value})") # Less verbose
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Buyers do not respond to sell requests."""
        return False


class KaplanSeller(BaseTrader):
    """
    Kaplan Seller Strategy: Tries to 'snipe' bids or improve asks slightly.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        # Note: is_buyer is False for sellers
        super().__init__(name, False, private_values, strategy="kaplan")
        self.logger = logging.getLogger(f'trader.{self.name}')
        self.profit_margin_pct_most = kwargs.get('profit_margin_pct_most', 0.01)
        self.profit_margin_pct_quote = kwargs.get('profit_margin_pct_quote', 0.01)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Kaplan Seller logic:
        1. If profitable to sell at current bid-1 (snipe), submit ask = bid-1.
        2. Else if no bid, improve current ask by 1 if possible.
        3. Else, do nothing (wait).
        """
        if not self.can_trade(): return None

        limit_price = self.get_next_value_cost() # Seller's minimum willingness to accept (cost)
        if limit_price is None: self.logger.warning("Can trade but get_next_value_cost returned None."); return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        target_ask = None

        # Condition 1: Snipe the Bid?
        # Check if there's a bid above our limit price
        if current_bid is not None and limit_price <= current_bid:
            # Potential profit exists. Determine ask price.
            # Interpretation: Offer slightly below the current bid to steal the deal.
            # RMP94 description for buyer was "bid equal to smaller of ask or value".
            # Analogous seller: "ask equal to larger of bid or cost". Let's use this.
            target_ask = max(limit_price, current_bid)

            # Safety check: Ensure target_ask < current_ask if one exists
            if current_ask is not None and target_ask >= current_ask:
                self.logger.debug(f"Potential snipe ask {target_ask} is not better than current ask {current_ask}. Waiting.")
                target_ask = None # Don't place non-improving ask
            else:
                self.logger.debug(f"Targeting snipe ask {target_ask} (based on CBID {current_bid}, LIMIT {limit_price})")

        # Condition 2: Improve Current Ask? (Only if not sniping)
        elif target_ask is None and current_ask is not None:
            # Try to improve the current ask by the smallest increment (e.g., 1 lower)
            if limit_price < current_ask: # Can we ask lower?
                potential_ask = current_ask - 1
                target_ask = max(limit_price, potential_ask) # Ask improvement, capped by limit (cost)
                self.logger.debug(f"Targeting improved ask {target_ask} (improving CASK {current_ask}, LIMIT {limit_price})")
            else:
                 self.logger.debug(f"Cannot improve current ask {current_ask} (LIMIT {limit_price})")

        # Condition 3: Initial Ask? (Only if not sniping and no current ask)
        elif target_ask is None and current_ask is None:
             # Default to 'pass' if market is empty or no opportunity.
             self.logger.debug("No bid to snipe and no ask to improve. Waiting.")
             target_ask = None

        # Final submission
        if target_ask is not None:
            # Ensure ask is within market bounds
            final_ask = max(self.min_price, min(self.max_price, int(round(target_ask))))
            self.logger.debug(f"Proposing final ask {final_ask}")
            return final_ask
        else:
            # Do not submit an ask this step
            self.logger.debug("Not submitting ask.")
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Sellers do not respond to buy requests."""
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Kaplan accepts any profitable bid."""
        if not self.can_trade() or current_bid_price is None:
            return False
        cost = self.get_next_value_cost()
        if cost is None:
            return False
        is_profitable = (current_bid_price >= cost)
        if is_profitable:
            self.logger.debug(f"Requesting SELL at {current_bid_price} (Cost={cost})")
        # else: self.logger.debug(f"Rejecting SELL at {current_bid_price} (Cost={cost})")
        return is_profitable
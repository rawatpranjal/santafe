# code/traders/kaplan.py
import random
import logging
from .base import BaseTrader

class KaplanBuyer(BaseTrader):
    """
    Kaplan Buyer Strategy: Tries to 'snipe' asks or improve bids slightly.
    Based on RMP94 description.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="kaplan")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Parameters (optional, defaults are placeholders, might not be needed for base Kaplan)
        # self.profit_margin_pct_most = kwargs.get('profit_margin_pct_most', 0.01)
        # self.profit_margin_pct_quote = kwargs.get('profit_margin_pct_quote', 0.01)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Kaplan Buyer logic based on RMP94 description:
        "places a bid equal to the smaller of the current ask or its current token value"
        Only if this improves the current bid.
        """
        if not self.can_trade(): return None
        limit_price = self.get_next_value_cost();
        if limit_price is None: self.logger.warning("Can trade but no limit price."); return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        target_bid = None

        # Condition 1: Snipe the Ask? (Primary Kaplan logic)
        if current_ask is not None:
            potential_bid = min(limit_price, current_ask)

            # Check if profitable AND improves current market bid
            if potential_bid >= self.min_price and (current_bid is None or potential_bid > current_bid):
                 target_bid = potential_bid
                 self.logger.debug(f"Targeting snipe bid {target_bid} (Ask={current_ask}, Limit={limit_price}, CBid={current_bid})")
            else:
                 self.logger.debug(f"Potential snipe {potential_bid} not profitable or not improving CBid={current_bid}. Waiting.")
        else:
            # No ask to react to. RMP94 Kaplan is reactive. Wait.
            self.logger.debug("No ask to snipe. Waiting.")

        # Final submission
        if target_bid is not None:
            final_bid = max(self.min_price, min(self.max_price, int(round(target_bid))))
            # self.logger.debug(f"Proposing final bid {final_bid}") # Reduce noise
            return final_bid
        else:
            # self.logger.debug("Not submitting bid.") # Reduce noise
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Kaplan accepts any profitable offer."""
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost();
        if value is None: return False
        is_profitable = (current_offer_price <= value)
        if is_profitable: self.logger.debug(f"Requesting BUY at {current_offer_price} (Value={value})")
        # Clear RL state if auto-accepting
        if is_profitable: self._current_step_state = None; self._current_step_action = None; self._current_step_log_prob = None; self._current_step_value = None;
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Buyers do not respond."""
        return False


class KaplanSeller(BaseTrader):
    """
    Kaplan Seller Strategy: Tries to 'snipe' bids or improve asks slightly.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="kaplan")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # self.profit_margin_pct_most = kwargs.get('profit_margin_pct_most', 0.01)
        # self.profit_margin_pct_quote = kwargs.get('profit_margin_pct_quote', 0.01)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Kaplan Seller logic (analogous to buyer):
        "places an ask equal to the larger of the current bid or its current token cost"
        Only if this improves the current ask.
        """
        if not self.can_trade(): return None
        limit_price = self.get_next_value_cost(); # Seller's cost
        if limit_price is None: self.logger.warning("Can trade but no limit price."); return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        target_ask = None

        # Condition 1: Snipe the Bid?
        if current_bid is not None:
            potential_ask = max(limit_price, current_bid)

            # Check if potentially profitable AND improves current market ask
            if potential_ask <= self.max_price and (current_ask is None or potential_ask < current_ask):
                target_ask = potential_ask
                self.logger.debug(f"Targeting snipe ask {target_ask} (CBid={current_bid}, Limit={limit_price}, CAsk={current_ask})")
            else:
                 self.logger.debug(f"Potential snipe {potential_ask} not profitable or not improving CAsk={current_ask}. Waiting.")
        else:
            # No bid to react to. Wait.
            self.logger.debug("No bid to snipe. Waiting.")

        # Final submission
        if target_ask is not None:
            final_ask = max(self.min_price, min(self.max_price, int(round(target_ask))))
            # self.logger.debug(f"Proposing final ask {final_ask}") # Reduce noise
            return final_ask
        else:
            # self.logger.debug("Not submitting ask.") # Reduce noise
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Sellers do not respond."""
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """Kaplan accepts any profitable bid."""
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost();
        if cost is None: return False
        is_profitable = (current_bid_price >= cost)
        if is_profitable: self.logger.debug(f"Requesting SELL at {current_bid_price} (Cost={cost})")
        # Clear RL state if auto-accepting
        if is_profitable: self._current_step_state = None; self._current_step_action = None; self._current_step_log_prob = None; self._current_step_value = None;
        return is_profitable
# traders/gd.py
import logging
import numpy as np
from collections import deque
from .base import BaseTrader

class GDBuyer(BaseTrader):
    """
    Gjerstad-Dickhaut (GD) Buyer Strategy (Simplified Beliefs):
    - History now persists across periods within a round.
    - Estimates probability of acceptance based on recent market asks in history.
    - Chooses bid price to maximize immediate expected profit.
    - Accepts offers if immediately profitable.
    """
    def __init__(self, name, is_buyer, private_values, history_len=50, **kwargs): # Increased default history
        super().__init__(name, True, private_values, strategy="gd")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if history_len <= 0: history_len = 1 # Ensure positive length
        self.history_len = history_len
        # Store recent market observations (ask prices relevant for buyer)
        self.market_ask_history = deque(maxlen=self.history_len)
        self.logger.debug(f"Initialized GD Buyer: history_len={self.history_len}")

    def _reset_learning_state(self):
        """ Clear history at the start of a new ROUND. """
        self.market_ask_history.clear()
        self.logger.debug("GD history cleared for new round.")

    def _update_history(self, current_ask_info):
        """ Add current best ask to history if it exists. """
        if current_ask_info and isinstance(current_ask_info.get('price'), (int, float)):
            # Only add if price is valid
            price = current_ask_info['price']
            if self.min_price <= price <= self.max_price:
                 self.market_ask_history.append(price)

    def _estimate_prob_accept(self, bid_price):
        """ Estimate P(accept | bid=bid_price) based on recent asks in round history. """
        if not self.market_ask_history or bid_price < self.min_price:
            return 0.0

        relevant_asks = np.array(list(self.market_ask_history))
        if len(relevant_asks) == 0: return 0.0

        # Probability = fraction of historical asks <= current potential bid
        accepting_asks_count = np.sum(relevant_asks <= bid_price)
        probability = accepting_asks_count / len(relevant_asks)
        return probability

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Update history, then choose bid to maximize expected surplus. """
        self._update_history(current_ask_info) # Update history based on current market state

        if not self.can_trade(): return None
        value = self.get_next_value_cost()
        if value is None: return None

        best_expected_profit = -1e-9 # Initialize slightly below zero
        best_bid = None

        # Iterate through potential profitable bids
        for potential_bid in range(self.min_price, value + 1):
            prob_accept = self._estimate_prob_accept(potential_bid)
            expected_profit = prob_accept * (value - potential_bid)

            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_bid = potential_bid

        # Only submit if expected profit is positive
        if best_bid is not None and best_expected_profit > 0:
            # self.logger.debug(f"GD proposing bid {best_bid} (Value={value}, EProfit={best_expected_profit:.2f})")
            return best_bid
        else:
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable (like ZIC). """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        is_profitable = (value is not None and current_offer_price <= value)
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def _clear_rl_step_state(self):
        self._current_step_state = None; self._current_step_action = None
        self._current_step_log_prob = None; self._current_step_value = None

class GDSeller(BaseTrader):
    """
    Gjerstad-Dickhaut (GD) Seller Strategy (Simplified Beliefs):
    - History persists across periods within a round.
    - Estimates probability of acceptance based on recent market bids.
    - Chooses ask price to maximize immediate expected profit.
    - Accepts bids if immediately profitable.
    """
    def __init__(self, name, is_buyer, private_values, history_len=50, **kwargs): # Increased default history
        super().__init__(name, False, private_values, strategy="gd")
        self.logger = logging.getLogger(f'trader.{self.name}')
        if history_len <= 0: history_len = 1
        self.history_len = history_len
        self.market_bid_history = deque(maxlen=self.history_len) # Store bid prices
        self.logger.debug(f"Initialized GD Seller: history_len={self.history_len}")

    def _reset_learning_state(self):
        """ Clear history at the start of a new ROUND. """
        self.market_bid_history.clear()
        self.logger.debug("GD history cleared for new round.")

    def _update_history(self, current_bid_info):
        """ Add current best bid to history if it exists and is valid. """
        if current_bid_info and isinstance(current_bid_info.get('price'), (int, float)):
            price = current_bid_info['price']
            if self.min_price <= price <= self.max_price:
                self.market_bid_history.append(price)

    def _estimate_prob_accept(self, ask_price):
        """ Estimate P(accept | ask=ask_price) based on recent bids. """
        if not self.market_bid_history or ask_price > self.max_price:
            return 0.0

        relevant_bids = np.array(list(self.market_bid_history))
        if len(relevant_bids) == 0: return 0.0

        # Probability = fraction of historical bids >= current potential ask
        accepting_bids_count = np.sum(relevant_bids >= ask_price)
        probability = accepting_bids_count / len(relevant_bids)
        return probability

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Update history, then choose ask to maximize expected surplus. """
        self._update_history(current_bid_info)

        if not self.can_trade(): return None
        cost = self.get_next_value_cost()
        if cost is None: return None

        best_expected_profit = -1e-9
        best_ask = None

        # Iterate through potential profitable asks
        for potential_ask in range(cost, self.max_price + 1):
            prob_accept = self._estimate_prob_accept(potential_ask)
            expected_profit = prob_accept * (potential_ask - cost)

            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_ask = potential_ask

        if best_ask is not None and best_expected_profit > 0:
            # self.logger.debug(f"GD proposing ask {best_ask} (Cost={cost}, EProfit={best_expected_profit:.2f})")
            return best_ask
        else:
            return None

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable (like ZIC). """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        is_profitable = (cost is not None and current_bid_price >= cost)
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def _clear_rl_step_state(self):
        self._current_step_state = None; self._current_step_action = None
        self._current_step_log_prob = None; self._current_step_value = None
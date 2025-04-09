# traders/el.py
import random
import logging
import numpy as np
from .base import BaseTrader

class ELBuyer(BaseTrader):
    """
    Easley-Ledyard (EL) Buyer Strategy:
    - Adjusts reservation price based on previous period's success/failure.
    - Bids based on current reservation price.
    - Accepts profitable offers based on current reservation price.
    """
    def __init__(self, name, is_buyer, private_values, initial_markup=0.05, adjustment_rate=0.1, **kwargs):
        super().__init__(name, True, private_values, strategy="el")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Store original values for reset
        self._original_private_values = list(private_values)
        # Current operative reservation prices (start same as private values)
        self.reservation_prices = list(private_values)
        # Track success/failure per token index within a round
        self.trade_success = [None] * self.max_tokens # None: not attempted, True: traded, False: failed/not traded
        self.adjustment_rate = adjustment_rate # How much to adjust reservation price
        self.logger.debug(f"Initialized EL Buyer: adjustment_rate={self.adjustment_rate}")

    def _reset_learning_state(self):
        """ Reset reservation prices and success tracker at the start of a new ROUND. """
        self.reservation_prices = list(self._original_private_values)
        self.trade_success = [None] * self.max_tokens
        self.logger.debug("EL state reset for new round.")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """ Reset only tokens_left. Keep learned reservation prices. """
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        # Mark remaining tokens as not attempted for this period
        start_idx = self.max_tokens - self.tokens_left
        for i in range(start_idx, self.max_tokens):
             self.trade_success[i] = None
        self.logger.debug(f"EL reset for period {period_idx}. Tokens={self.tokens_left}. Reservations: {self.reservation_prices[:self.max_tokens]}")

    def _get_current_reservation_price(self):
        """ Get the current reservation price for the next available token. """
        if not self.can_trade():
            return None
        idx = self.max_tokens - self.tokens_left
        if 0 <= idx < len(self.reservation_prices):
            return self.reservation_prices[idx]
        else:
            self.logger.error(f"Invalid index {idx} accessing reservation_prices")
            return None

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Propose a bid slightly below the current reservation price. """
        if not self.can_trade(): return None
        res_price = self._get_current_reservation_price()
        if res_price is None: return None

        # Simple heuristic: bid slightly below reservation price
        # More complex EL might condition this on market state/time
        bid_price = int(round(res_price * 0.98)) # Example: bid 98% of reservation
        bid_price = max(self.min_price, min(self.max_price, bid_price))

        # Ensure bid <= reservation price after rounding/clamping
        bid_price = min(bid_price, res_price)
        bid_price = max(self.min_price, bid_price) # Ensure >= min_price

        if bid_price < self.min_price: return None # Cannot bid below min price

        # self.logger.debug(f"EL proposing bid {bid_price} (Res={res_price})")
        return bid_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if offer is below current reservation price. """
        if not self.can_trade() or current_offer_price is None: return False
        res_price = self._get_current_reservation_price()
        is_profitable = (res_price is not None and current_offer_price <= res_price)
        # if is_profitable: self.logger.debug(f"EL accepting ask {current_offer_price} (Res={res_price})")
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def record_trade(self, period, step, price):
        """ Mark token as successfully traded. """
        idx = self.max_tokens - self.tokens_left
        profit = super().record_trade(period, step, price)
        if profit is not None and 0 <= idx < self.max_tokens:
             self.trade_success[idx] = True # Mark as success
        return profit

    def update_end_of_period(self, period_stats):
        """ Adjust reservation prices based on success/failure in the finished period. """
        self.logger.debug(f"EL updating reservations at end of P{self.current_period}")
        for i in range(self.max_tokens):
            # Adjust only if the token wasn't traded successfully (or attempt failed)
            if self.trade_success[i] is None or self.trade_success[i] is False:
                original_value = self._original_private_values[i]
                current_res = self.reservation_prices[i]
                # Become more aggressive: increase reservation price (bid higher next time)
                # Move towards the original value
                adjustment = self.adjustment_rate * (original_value - current_res)
                new_res = current_res + adjustment
                # Clamp adjustment: don't go above original value or below min price
                self.reservation_prices[i] = int(round(max(self.min_price, min(original_value, new_res))))
                self.logger.debug(f"  Token {i} not traded. Adjusting Res: {current_res} -> {self.reservation_prices[i]} (Orig={original_value})")
            # else: If self.trade_success[i] is True, maybe become slightly less aggressive?
                 # current_res = self.reservation_prices[i]
                 # adjustment = - self.adjustment_rate * 0.1 * current_res # Small decrease
                 # self.reservation_prices[i] = int(round(max(self.min_price, current_res + adjustment)))

class ELSeller(BaseTrader):
    """
    Easley-Ledyard (EL) Seller Strategy:
    - Adjusts reservation price (cost) based on previous period's success/failure.
    - Asks based on current reservation price.
    - Accepts profitable bids based on current reservation price.
    """
    def __init__(self, name, is_buyer, private_values, initial_markup=0.05, adjustment_rate=0.1, **kwargs):
        super().__init__(name, False, private_values, strategy="el")
        self.logger = logging.getLogger(f'trader.{self.name}')
        self._original_private_values = list(private_values)
        self.reservation_prices = list(private_values) # Cost here
        self.trade_success = [None] * self.max_tokens
        self.adjustment_rate = adjustment_rate
        self.logger.debug(f"Initialized EL Seller: adjustment_rate={self.adjustment_rate}")

    def _reset_learning_state(self):
        """ Reset reservation prices and success tracker at the start of a new ROUND. """
        self.reservation_prices = list(self._original_private_values)
        self.trade_success = [None] * self.max_tokens
        self.logger.debug("EL state reset for new round.")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        start_idx = self.max_tokens - self.tokens_left
        for i in range(start_idx, self.max_tokens):
             self.trade_success[i] = None
        self.logger.debug(f"EL reset for period {period_idx}. Tokens={self.tokens_left}. Reservations: {self.reservation_prices[:self.max_tokens]}")

    def _get_current_reservation_price(self):
        """ Get the current reservation cost for the next available token. """
        if not self.can_trade(): return None
        idx = self.max_tokens - self.tokens_left
        if 0 <= idx < len(self.reservation_prices):
            return self.reservation_prices[idx]
        else: self.logger.error(f"Invalid index {idx} accessing reservation_prices"); return None

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Propose an ask slightly above the current reservation cost. """
        if not self.can_trade(): return None
        res_price = self._get_current_reservation_price() # This is the cost
        if res_price is None: return None

        # Ask slightly above reservation cost
        ask_price = int(round(res_price * 1.02)) # Example: ask 102% of reservation
        ask_price = max(self.min_price, min(self.max_price, ask_price))

        # Ensure ask >= reservation price after rounding/clamping
        ask_price = max(ask_price, res_price)
        ask_price = min(self.max_price, ask_price) # Ensure <= max_price

        if ask_price > self.max_price: return None

        # self.logger.debug(f"EL proposing ask {ask_price} (Res={res_price})")
        return ask_price

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if bid is above current reservation cost. """
        if not self.can_trade() or current_bid_price is None: return False
        res_price = self._get_current_reservation_price()
        is_profitable = (res_price is not None and current_bid_price >= res_price)
        # if is_profitable: self.logger.debug(f"EL accepting bid {current_bid_price} (Res={res_price})")
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def record_trade(self, period, step, price):
        idx = self.max_tokens - self.tokens_left
        profit = super().record_trade(period, step, price)
        if profit is not None and 0 <= idx < self.max_tokens:
             self.trade_success[idx] = True
        return profit

    def update_end_of_period(self, period_stats):
        """ Adjust reservation costs based on success/failure in the finished period. """
        self.logger.debug(f"EL updating reservations at end of P{self.current_period}")
        for i in range(self.max_tokens):
            if self.trade_success[i] is None or self.trade_success[i] is False:
                original_cost = self._original_private_values[i]
                current_res = self.reservation_prices[i]
                # Become more aggressive: decrease reservation cost (ask lower next time)
                # Move towards the original cost
                adjustment = self.adjustment_rate * (original_cost - current_res)
                new_res = current_res + adjustment
                # Clamp adjustment: don't go below original cost or above max price
                self.reservation_prices[i] = int(round(min(self.max_price, max(original_cost, new_res))))
                self.logger.debug(f"  Token {i} not traded. Adjusting Res: {current_res} -> {self.reservation_prices[i]} (Orig={original_cost})")
            # else: If traded, maybe become less aggressive (raise reservation cost slightly)?
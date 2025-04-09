# traders/kp.py
import logging
from .base import BaseTrader
import numpy as np

class KaplanBuyer(BaseTrader):
    """
    Kaplan Buyer Strategy (Based on RPM paper description):
    - Waits in background by default.
    - Snipes by bidding the current ask if profitable and improving.
    - Switches to "truthtelling" mode based on inactivity/time remaining.
    - Truthtelling mode bids min(ask, value - 1).
    - Accepts profitable trades.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="kaplan")
        self.logger = logging.getLogger(f'trader.{self.name}')

        # Parameters for switching to truthtelling mode (from RPM p.32)
        # These might need tuning or be sourced from elsewhere if available
        self.tt_inactive_market_thresh = kwargs.get('kp_tt_inactive_market', 5) # Steps since market trade
        self.tt_inactive_self_frac = kwargs.get('kp_tt_inactive_self_frac', 2.0/3.0) # Fraction of remaining time since own trade
        # Thresholds for "long time" or "time running out" are less clear, maybe combine?
        # Let's add a simple remaining steps threshold as well
        self.tt_time_running_out_steps = kwargs.get('kp_tt_time_running_out', 5) # e.g., last 5 steps

        # Internal State
        self.mode = 'background' # 'background' or 'truthtelling'
        self.steps_since_last_market_trade = 0
        self.steps_since_own_last_trade = 0

        self.logger.debug(f"Initialized KP Buyer.")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """ Reset period-specific state. """
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.mode = 'background' # Start in background mode
        self.steps_since_last_market_trade = 0
        self.steps_since_own_last_trade = 0
        self._last_market_trade_step = -1 # Track last step trade occurred

    def record_trade(self, period, step, price):
        """ Reset own inactivity timer when own trade occurs. """
        profit = super().record_trade(period, step, price)
        if profit is not None:
            self.steps_since_own_last_trade = 0
            self._last_market_trade_step = step # Also update market timer on own trade
        return profit

    def _check_and_update_state(self, market_history):
        """ Update timers and potentially switch mode. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        last_trade_info = market_history.get('last_trade_info_for_period')
        last_trade_step = -1
        if last_trade_info:
            last_trade_step = last_trade_info.get('step', -1)

        # Increment timers if no trade occurred in the *previous* step
        if last_trade_step > self._last_market_trade_step :
             self.steps_since_last_market_trade = self.current_step - last_trade_step
             self._last_market_trade_step = last_trade_step
        else:
             self.steps_since_last_market_trade += 1

        # Increment own timer (always increments unless reset by record_trade)
        self.steps_since_own_last_trade += 1

        # --- Check conditions for switching to Truthtelling ---
        if self.mode == 'background':
            remaining_steps = self.total_steps_in_period - self.current_step - 1 # Steps remaining *after* this one
            if remaining_steps < 0: remaining_steps = 0

            # Condition from RPM p32 (complex one first)
            cond_complex = (self.steps_since_last_market_trade >= self.tt_inactive_market_thresh and
                            self.steps_since_own_last_trade > self.tt_inactive_self_frac * remaining_steps)

            # Simpler condition: Time running out
            cond_time_out = (remaining_steps < self.tt_time_running_out_steps)

            # Combine conditions (OR logic based on text)
            # RPM is slightly ambiguous if the '5 steps since last transaction' is part of the complex condition
            # or a separate trigger. Let's assume it's part of the complex one as written.
            if cond_complex or cond_time_out:
                self.logger.debug(f"{step_id}: KP Switching to Truthtelling mode. "
                                  f"MarketInactive={self.steps_since_last_market_trade}, "
                                  f"SelfInactive={self.steps_since_own_last_trade}, "
                                  f"Remaining={remaining_steps}, ComplexTrig={cond_complex}, TimeOutTrig={cond_time_out}")
                self.mode = 'truthtelling'
        # Note: RPM doesn't specify switching *back* from truthtelling within a period.
        # We assume it stays in truthtelling once triggered until the next period.

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Decide action based on current mode (Background/Truthtelling). """
        self._check_and_update_state(market_history) # Update timers and mode

        if not self.can_trade(): return None
        value = self.get_next_value_cost()
        if value is None: return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"

        target_bid = None
        if self.mode == 'background':
            # --- Background Mode: Snipe ---
            # "places a bid equal to the smaller of the current ask or its current token value"
            # Interpretation: Bid 'ask' only if ask < value and it improves current bid
            if current_ask is not None:
                try:
                    ask_f = float(current_ask)
                    if ask_f < value: # Can potentially make profit by matching ask
                        potential_bid = ask_f # Target is the ask itself
                        is_improving = (current_bid is None or potential_bid > float(current_bid))
                        if potential_bid >= self.min_price and is_improving:
                            target_bid = potential_bid
                            self.logger.debug(f"{step_id}: KP (Background) targeting snipe bid {target_bid} (Ask={ask_f} < Value={value}, improves CBid={current_bid})")
                        # else: self.logger.debug(f"{step_id}: KP (Background) snipe {potential_bid} doesn't improve CBid {current_bid}.")
                    # else: self.logger.debug(f"{step_id}: KP (Background) Ask {ask_f} >= Value {value}, cannot snipe.")
                except (ValueError, TypeError) as e:
                     self.logger.warning(f"{step_id}: KP (Background) error during snipe check: {e}")
            # else: self.logger.debug(f"{step_id}: KP (Background) no ask to snipe.")

        elif self.mode == 'truthtelling':
            # --- Truthtelling Mode ---
            # "places a bid equal to the minimum of the current ask and T-1"
            if current_ask is not None:
                try:
                    ask_f = float(current_ask)
                    # Bid just under value, but no higher than the ask
                    potential_bid = min(value - 1, ask_f)
                    target_bid = potential_bid # Always try to bid in truthtelling mode? Assume yes.
                    self.logger.debug(f"{step_id}: KP (Truthtelling) targeting bid {target_bid} = min(Ask={ask_f}, Value-1={value-1})")
                except (ValueError, TypeError) as e:
                     self.logger.warning(f"{step_id}: KP (Truthtelling) error during bid calc: {e}")
                     target_bid = None
            else: # No ask? Maybe bid value-1? Or wait? Let's bid value-1.
                 target_bid = value - 1
                 self.logger.debug(f"{step_id}: KP (Truthtelling) no ask, targeting bid {target_bid} (Value-1)")

        # --- Finalize and Submit ---
        if target_bid is not None:
            try:
                # Clamp to market bounds and ensure profitability (value >= bid)
                final_bid = max(self.min_price, min(self.max_price, int(round(target_bid))))
                final_bid = min(final_bid, value) # Ensure doesn't bid > value
                final_bid = max(self.min_price, final_bid) # Ensure >= min_price

                # In background mode, re-check improvement *after* clamping
                if self.mode == 'background':
                    is_improving_final = (current_bid is None or final_bid > float(current_bid))
                    if not is_improving_final:
                        # self.logger.debug(f"{step_id}: KP (Background) final bid {final_bid} doesn't improve CBid={current_bid}. Holding.")
                        return None # Don't submit non-improving background snipe

                self.logger.debug(f"{step_id}: KP ({self.mode}) proposing final bid {final_bid}.")
                return final_bid
            except (ValueError, TypeError) as e:
                self.logger.warning(f"{step_id}: KP Error finalizing bid: {e}")
                return None
        else:
            # No target bid was set
            # self.logger.debug(f"{step_id}: KP ({self.mode}) No target bid calculated.")
            return None


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs value. """
        if not self.can_trade() or current_offer_price is None: return False
        value = self.get_next_value_cost()
        if value is None: return False
        try: is_profitable = (float(current_offer_price) <= value)
        except (ValueError, TypeError): return False
        if is_profitable: self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False


class KPSeller(BaseTrader):
    """
    Kaplan Seller Strategy (Mirrors Buyer):
    - Waits in background by default.
    - Snipes by asking the current bid if profitable and improving.
    - Switches to "truthtelling" mode based on inactivity/time remaining.
    - Truthtelling mode asks max(bid, cost + 1).
    - Accepts profitable trades.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="kaplan") # is_buyer=False
        self.logger = logging.getLogger(f'trader.{self.name}')

        # Parameters for switching modes (mirror buyer)
        self.tt_inactive_market_thresh = kwargs.get('kp_tt_inactive_market', 5)
        self.tt_inactive_self_frac = kwargs.get('kp_tt_inactive_self_frac', 2.0/3.0)
        self.tt_time_running_out_steps = kwargs.get('kp_tt_time_running_out', 5)

        # Internal State
        self.mode = 'background'
        self.steps_since_last_market_trade = 0
        self.steps_since_own_last_trade = 0
        self._last_market_trade_step = -1

        self.logger.debug(f"Initialized KP Seller.")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.mode = 'background'
        self.steps_since_last_market_trade = 0
        self.steps_since_own_last_trade = 0
        self._last_market_trade_step = -1

    def record_trade(self, period, step, price):
        profit = super().record_trade(period, step, price)
        if profit is not None:
            self.steps_since_own_last_trade = 0
            self._last_market_trade_step = step
        return profit

    # _check_and_update_state is identical to KPBuyer's version
    def _check_and_update_state(self, market_history):
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        last_trade_info = market_history.get('last_trade_info_for_period')
        last_trade_step = -1
        if last_trade_info: last_trade_step = last_trade_info.get('step', -1)

        if last_trade_step > self._last_market_trade_step :
             self.steps_since_last_market_trade = self.current_step - last_trade_step
             self._last_market_trade_step = last_trade_step
        else: self.steps_since_last_market_trade += 1
        self.steps_since_own_last_trade += 1

        if self.mode == 'background':
            remaining_steps = self.total_steps_in_period - self.current_step - 1
            if remaining_steps < 0: remaining_steps = 0
            cond_complex = (self.steps_since_last_market_trade >= self.tt_inactive_market_thresh and
                            self.steps_since_own_last_trade > self.tt_inactive_self_frac * remaining_steps)
            cond_time_out = (remaining_steps < self.tt_time_running_out_steps)
            if cond_complex or cond_time_out:
                self.logger.debug(f"{step_id}: KP Switching to Truthtelling mode. "
                                  f"MarketInactive={self.steps_since_last_market_trade}, "
                                  f"SelfInactive={self.steps_since_own_last_trade}, "
                                  f"Remaining={remaining_steps}, ComplexTrig={cond_complex}, TimeOutTrig={cond_time_out}")
                self.mode = 'truthtelling'

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Decide action based on current mode (Background/Truthtelling). """
        self._check_and_update_state(market_history)

        if not self.can_trade(): return None
        cost = self.get_next_value_cost()
        if cost is None: return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"

        target_ask = None
        if self.mode == 'background':
            # --- Background Mode: Snipe ---
            # Seller equivalent: Ask 'bid' only if bid > cost and it improves current ask
            if current_bid is not None:
                try:
                    bid_f = float(current_bid)
                    if bid_f > cost: # Can potentially make profit by matching bid
                        potential_ask = bid_f # Target is the bid itself
                        is_improving = (current_ask is None or potential_ask < float(current_ask))
                        if potential_ask <= self.max_price and is_improving:
                            target_ask = potential_ask
                            self.logger.debug(f"{step_id}: KP (Background) targeting snipe ask {target_ask} (Bid={bid_f} > Cost={cost}, improves CAsk={current_ask})")
                        # else: self.logger.debug(f"{step_id}: KP (Background) snipe {potential_ask} doesn't improve CAsk {current_ask}.")
                    # else: self.logger.debug(f"{step_id}: KP (Background) Bid {bid_f} <= Cost {cost}, cannot snipe.")
                except (ValueError, TypeError) as e:
                     self.logger.warning(f"{step_id}: KP (Background) error during snipe check: {e}")
            # else: self.logger.debug(f"{step_id}: KP (Background) no bid to snipe.")

        elif self.mode == 'truthtelling':
            # --- Truthtelling Mode ---
            # Seller equivalent: Ask max(bid, cost + 1) ??
            # RPM p32 only describes buyer bid T-1. Seller logic isn't explicit.
            # Let's assume seller asks cost+1, but no lower than current bid.
            potential_ask = cost + 1
            if current_bid is not None:
                 try: potential_ask = max(cost + 1, float(current_bid))
                 except (ValueError, TypeError): pass # Keep cost+1 if bid invalid
            target_ask = potential_ask
            self.logger.debug(f"{step_id}: KP (Truthtelling) targeting ask {target_ask} (Based on Cost+1={cost+1}, CBid={current_bid})")


        # --- Finalize and Submit ---
        if target_ask is not None:
            try:
                final_ask = max(self.min_price, min(self.max_price, int(round(target_ask))))
                final_ask = max(final_ask, cost) # Ensure profitable (ask >= cost)
                final_ask = min(self.max_price, final_ask) # Ensure <= max_price

                # In background mode, re-check improvement *after* clamping
                if self.mode == 'background':
                    is_improving_final = (current_ask is None or final_ask < float(current_ask))
                    if not is_improving_final:
                        # self.logger.debug(f"{step_id}: KP (Background) final ask {final_ask} doesn't improve CAsk={current_ask}. Holding.")
                        return None

                self.logger.debug(f"{step_id}: KP ({self.mode}) proposing final ask {final_ask}.")
                return final_ask
            except (ValueError, TypeError) as e:
                 self.logger.warning(f"{step_id}: KP Error finalizing ask: {e}")
                 return None
        else:
            # self.logger.debug(f"{step_id}: KP ({self.mode}) No target ask calculated.")
            return None


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs cost. """
        if not self.can_trade() or current_bid_price is None: return False
        cost = self.get_next_value_cost()
        if cost is None: return False
        try: is_profitable = (float(current_bid_price) >= cost)
        except (ValueError, TypeError): return False
        if is_profitable: self._clear_rl_step_state()
        return is_profitable
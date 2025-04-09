# traders/rg.py
import logging
from .base import BaseTrader
import numpy as np # Import numpy for isnan check

class RGBuyer(BaseTrader):
    """
    Ringuette Buyer Strategy: Waits until bid >= ask - margin, then acts.
    Needs state to act only once per period.
    Interpretation: Once condition met, act like Kaplan (snipe).
    """
    def __init__(self, name, is_buyer, private_values, profit_margin=5, **kwargs): # Example default margin
        super().__init__(name, True, private_values, strategy="rg")
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Ensure margin is numeric and sensible
        try:
            self.profit_margin = float(profit_margin)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid profit_margin '{profit_margin}'. Using default 5.")
            self.profit_margin = 5.0
        self.acted_this_period = False
        self.logger.debug(f"Initialized RG Buyer with margin={self.profit_margin:.2f}")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """ Reset period-specific state including the acted flag. """
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.acted_this_period = False
        # self.logger.debug(f"RG Buyer acted_this_period flag reset for P{period_idx}.") # Optional: reduce log noise

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ If condition met and haven't acted, submit Kaplan-like snipe bid. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}" # For logging context

        if not self.can_trade():
            # self.logger.debug(f"{step_id}: Cannot trade (Tokens left: {self.tokens_left}).")
            return None
        if self.acted_this_period:
            # self.logger.debug(f"{step_id}: Already acted this period.")
            return None

        value = self.get_next_value_cost()
        if value is None:
            self.logger.warning(f"{step_id}: Cannot get value despite can_trade=True.")
            return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        # --- Ringuette Condition Check with Debugging ---
        condition_met = False
        log_msg = f"{step_id}: Checking RG condition. CBid={current_bid}, CAsk={current_ask}, Margin={self.profit_margin}. "
        if current_bid is not None and current_ask is not None:
            try:
                 # Ensure values are numeric before comparison
                 bid_f = float(current_bid)
                 ask_f = float(current_ask)
                 margin_f = float(self.profit_margin)
                 if bid_f >= (ask_f - margin_f):
                     condition_met = True
                     log_msg += f"Condition MET: {bid_f:.2f} >= {ask_f:.2f} - {margin_f:.2f}"
                 else:
                     log_msg += f"Condition NOT MET: {bid_f:.2f} < {ask_f:.2f} - {margin_f:.2f}"
            except (ValueError, TypeError) as e:
                 log_msg += f"Error during condition check: {e}"
                 self.logger.warning(log_msg) # Log error
                 condition_met = False # Assume false on error
        else:
            log_msg += "Condition NOT MET (missing bid or ask)."

        self.logger.debug(log_msg) # Log condition check result regardless

        if not condition_met:
            return None # Wait if condition not met

        # --- Condition met: Attempt Kaplan-like Snipe ---
        self.logger.debug(f"{step_id}: Condition met, attempting snipe. Value={value}")
        target_bid = None
        if current_ask is not None:
            try:
                ask_f = float(current_ask)
                # Willing to pay up to ask or own value, whichever is lower
                potential_bid = min(value, ask_f)
                self.logger.debug(f"{step_id}: Potential snipe bid = min({value}, {ask_f}) = {potential_bid}")

                # Check profitability and if it improves market best bid
                is_improving = (current_bid is None or potential_bid > float(current_bid))
                if potential_bid >= self.min_price and is_improving:
                     target_bid = potential_bid
                     self.logger.debug(f"{step_id}: Snipe viable (improves CBid={current_bid}). Target={target_bid}")
                else:
                     self.logger.debug(f"{step_id}: Snipe not viable (profitable but doesn't improve CBid={current_bid}).")
            except (ValueError, TypeError) as e:
                 self.logger.warning(f"{step_id}: Error checking snipe viability: {e}")
                 target_bid = None
        else:
            self.logger.debug(f"{step_id}: No ask exists, cannot snipe.")
            target_bid = None # Cannot snipe if no ask exists

        # --- Determine final action ---
        final_bid = None
        if target_bid is not None:
            try:
                # Clamp to market bounds and ensure integer price
                final_bid_calc = max(self.min_price, min(self.max_price, int(round(target_bid))))
                # Ensure still profitable vs own value after rounding/clamping
                final_bid_calc = min(final_bid_calc, value)
                final_bid_calc = max(self.min_price, final_bid_calc) # Final min price check

                # Re-check improvement *after* all clamping
                is_improving_final = (current_bid is None or final_bid_calc > float(current_bid))
                if is_improving_final:
                     final_bid = final_bid_calc
                     self.logger.debug(f"{step_id}: Proposing final snipe bid {final_bid}.")
                else:
                     self.logger.debug(f"{step_id}: Clamped/rounded snipe bid {final_bid_calc} no longer improves CBid {current_bid}. Holding.")
                     # If snipe not possible after all, should we still mark as acted? Let's say yes.
            except (ValueError, TypeError) as e:
                 self.logger.warning(f"{step_id}: Error finalizing snipe bid: {e}")
                 final_bid = None

        # Mark as acted *if the condition was met*, regardless of whether a bid was submitted
        self.logger.debug(f"{step_id}: Setting acted_this_period=True (condition was met).")
        self.acted_this_period = True
        return final_bid # Return the calculated bid or None


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs value, only if haven't acted. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        if not self.can_trade() or self.acted_this_period: return False
        if current_offer_price is None: return False

        value = self.get_next_value_cost()
        if value is None: return False

        try:
             offer_f = float(current_offer_price)
             is_profitable = (offer_f <= value)
        except (ValueError, TypeError):
             self.logger.warning(f"{step_id}: Invalid offer price {current_offer_price} in request_buy.")
             return False

        if is_profitable:
            self.logger.debug(f"{step_id}: RG accepting BUY at {current_offer_price} (Value={value}). Setting acted=True.")
            self.acted_this_period = True # Mark as acted if accepting
            self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Buyers don't accept sell requests

class RGSeller(BaseTrader):
    """
    Ringuette Seller Strategy: Waits until bid >= ask - margin, then acts.
    Interpretation: Once condition met, act like Kaplan (snipe).
    """
    def __init__(self, name, is_buyer, private_values, profit_margin=5, **kwargs): # Example default margin
        super().__init__(name, False, private_values, strategy="rg")
        self.logger = logging.getLogger(f'trader.{self.name}')
        try:
            self.profit_margin = float(profit_margin)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid profit_margin '{profit_margin}'. Using default 5.")
            self.profit_margin = 5.0
        self.acted_this_period = False
        self.logger.debug(f"Initialized RG Seller with margin={self.profit_margin:.2f}")

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """ Reset period-specific state including the acted flag. """
        super().reset_for_new_period(total_steps_in_period, round_idx, period_idx)
        self.acted_this_period = False
        # self.logger.debug(f"RG Seller acted_this_period flag reset for P{period_idx}.") # Optional: reduce log noise

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ If condition met and haven't acted, submit Kaplan-like snipe ask. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"

        if not self.can_trade() or self.acted_this_period: return None

        cost = self.get_next_value_cost()
        if cost is None:
            self.logger.warning(f"{step_id}: Cannot get cost despite can_trade=True.")
            return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        # --- Ringuette Condition Check with Debugging ---
        condition_met = False
        log_msg = f"{step_id}: Checking RG condition. CBid={current_bid}, CAsk={current_ask}, Margin={self.profit_margin}. "
        if current_bid is not None and current_ask is not None:
            try:
                 bid_f = float(current_bid)
                 ask_f = float(current_ask)
                 margin_f = float(self.profit_margin)
                 if bid_f >= (ask_f - margin_f):
                     condition_met = True
                     log_msg += f"Condition MET: {bid_f:.2f} >= {ask_f:.2f} - {margin_f:.2f}"
                 else:
                     log_msg += f"Condition NOT MET: {bid_f:.2f} < {ask_f:.2f} - {margin_f:.2f}"
            except (ValueError, TypeError) as e:
                 log_msg += f"Error during condition check: {e}"
                 self.logger.warning(log_msg)
                 condition_met = False
        else:
            log_msg += "Condition NOT MET (missing bid or ask)."

        self.logger.debug(log_msg)

        if not condition_met:
            return None

        # --- Condition met: Attempt Kaplan-like Snipe ---
        self.logger.debug(f"{step_id}: Condition met, attempting snipe. Cost={cost}")
        target_ask = None
        if current_bid is not None:
            try:
                bid_f = float(current_bid)
                # Willing to ask at least own cost, or potentially match/undercut bid
                potential_ask = max(cost, bid_f) # Simple version: match bid if profitable
                self.logger.debug(f"{step_id}: Potential snipe ask = max({cost}, {bid_f}) = {potential_ask}")

                # Check profitability and if it improves market best ask
                is_improving = (current_ask is None or potential_ask < float(current_ask))
                if potential_ask <= self.max_price and is_improving:
                     target_ask = potential_ask
                     self.logger.debug(f"{step_id}: Snipe viable (improves CAsk={current_ask}). Target={target_ask}")
                else:
                     self.logger.debug(f"{step_id}: Snipe not viable (profitable but doesn't improve CAsk={current_ask}).")
            except (ValueError, TypeError) as e:
                 self.logger.warning(f"{step_id}: Error checking snipe viability: {e}")
                 target_ask = None
        else:
            self.logger.debug(f"{step_id}: No bid exists, cannot snipe.")
            target_ask = None

        # --- Determine final action ---
        final_ask = None
        if target_ask is not None:
             try:
                final_ask_calc = max(self.min_price, min(self.max_price, int(round(target_ask))))
                final_ask_calc = max(final_ask_calc, cost) # Ensure profitable vs cost
                final_ask_calc = min(self.max_price, final_ask_calc) # Final max price check

                # Re-check improvement after clamping
                is_improving_final = (current_ask is None or final_ask_calc < float(current_ask))
                if is_improving_final:
                     final_ask = final_ask_calc
                     self.logger.debug(f"{step_id}: Proposing final snipe ask {final_ask}.")
                else:
                     self.logger.debug(f"{step_id}: Clamped/rounded snipe ask {final_ask_calc} no longer improves CAsk {current_ask}. Holding.")
             except (ValueError, TypeError) as e:
                 self.logger.warning(f"{step_id}: Error finalizing snipe ask: {e}")
                 final_ask = None

        # Mark as acted if condition was met
        self.logger.debug(f"{step_id}: Setting acted_this_period=True (condition was met).")
        self.acted_this_period = True
        return final_ask


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Sellers don't accept buy requests

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs cost, only if haven't acted. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        if not self.can_trade() or self.acted_this_period: return False
        if current_bid_price is None: return False

        cost = self.get_next_value_cost()
        if cost is None: return False

        try:
            bid_f = float(current_bid_price)
            is_profitable = (bid_f >= cost)
        except (ValueError, TypeError):
             self.logger.warning(f"{step_id}: Invalid bid price {current_bid_price} in request_sell.")
             return False

        if is_profitable:
            self.logger.debug(f"{step_id}: RG accepting SELL at {current_bid_price} (Cost={cost}). Setting acted=True.")
            self.acted_this_period = True
            self._clear_rl_step_state()
        return is_profitable
# traders/rg.py
import logging
from .base import BaseTrader
import numpy as np
import random # For potential random overbid logic

# Import the assumed Skeleton agent to use its logic
# Ensure sk.py exists and defines these classes
try:
    from .sk import SKBuyer as SkeletonBuyerLogic
    from .sk import SKSeller as SkeletonSellerLogic
except ImportError:
    logging.error("Could not import Skeleton logic from traders.sk. RG agent requires it.")
    # Define dummy classes to prevent NameError, but log a critical warning
    class SkeletonBuyerLogic: pass
    class SkeletonSellerLogic: pass
    logging.critical("Using dummy Skeleton logic for RG agent. SK implementation missing!")


class RGBuyer(BaseTrader):
    """
    Ringuette Buyer Strategy (Based on RPM paper description):
    - Waits in background initially.
    - Snipes by "randomly overbidding" ask (interpretation needed).
    - Switches to Skeleton mode based on inactivity timer.
    - Accepts profitable trades.
    - Revised: Only sets 'acted' flag if a quote is submitted or a trade accepted.
    - Revised: Fixed KeyError and IndexError in helper logic.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, True, private_values, strategy="rg")
        self.logger = logging.getLogger(f'trader.{self.name}')

        # Parameters for switching modes
        self.switch_time_threshold = kwargs.get('rg_switch_time', 12) # Steps since last trade threshold
        self.switch_remain_frac = kwargs.get('rg_switch_frac', 0.6)  # Fraction of remaining time threshold
        # Parameters for skeleton mode (if using assumed SK)
        self.sk_margin = kwargs.get('rg_sk_margin', 0.05)
        self.sk_shout_prob = kwargs.get('rg_sk_shout_prob', 0.9)
        # Parameter for overbidding (interpretation - needs tuning)
        self.overbid_amount = kwargs.get('rg_overbid', 1) # Simple +1 overbid

        # Internal State
        self.mode = 'background' # 'background' or 'skeleton'
        self.steps_since_last_market_trade = 0 # Track market inactivity
        self.acted_this_period = False # Has made a successful action (quote/accept)
        self._last_recorded_trade_step = -1 # Initialize timer tracking variable

        # Instantiate skeleton logic helper (uses assumed SK)
        try:
            # Ensure SkeletonBuyerLogic is the correct class name from sk.py
            self._sk_logic = SkeletonBuyerLogic(f"{name}_sk_helper", True, [0],
                                                 fixed_margin=self.sk_margin,
                                                 shout_probability=self.sk_shout_prob)
            self._sk_logic.update_market_params(self.min_price, self.max_price)
        except NameError: # If Skeleton logic couldn't be imported
            self._sk_logic = None # Set to None to indicate it's unavailable
            self.logger.error("RG Buyer could not instantiate SkeletonLogic helper.")


        self.logger.debug(f"Initialized RG Buyer: SwitchT={self.switch_time_threshold}, SwitchFrac={self.switch_remain_frac}, Overbid={self.overbid_amount}")

    def reset_for_new_period(self, round_idx, period_idx):
        """ Reset period-specific state including the acted flag. """
        super().reset_for_new_period(round_idx, period_idx)
        self.mode = 'background' # Start in background mode
        self.steps_since_last_market_trade = 0 # Reset timer
        self.acted_this_period = False
        self._last_recorded_trade_step = -1 # Reset timer tracker
        # Update total steps for helper agent if it exists
        if self._sk_logic:
            self._sk_logic.total_steps_in_period = total_steps_in_period
            self._sk_logic.current_period = period_idx
            self._sk_logic.current_round = round_idx
        # Ensure base class resets necessary fields like tokens_left, current_step
        # self.logger.debug(f"RG Buyer {self.name} reset for P{period_idx}. Acted={self.acted_this_period}, Mode={self.mode}")


    def record_trade(self, period, step, price):
        """ Reset own inactivity timer when own trade occurs. """
        profit = super().record_trade(period, step, price)
        # Note: We don't reset market timer here based on own trade,
        # only the _last_recorded_trade_step gets updated in _check_and_update_mode
        return profit

    # --- CORRECTED _check_and_update_mode ---
    def _check_and_update_mode(self, market_history):
        """ Check inactivity timers and switch mode if needed. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        last_trade_info = market_history.get('last_trade_info_for_period') # Use .get() for safety

        # Check if a trade happened *in the previous step or earlier*
        # We compare the current step with the step recorded in the last trade info
        trade_occurred_recently = False
        if last_trade_info and isinstance(last_trade_info, dict) and 'step' in last_trade_info:
            last_trade_step = last_trade_info['step']
            # Check if this trade step is newer than the last one we recorded
            if last_trade_step > self._last_recorded_trade_step:
                # A new trade happened since we last checked
                # Reset counter based on current step and last trade step
                self.steps_since_last_market_trade = self.current_step - last_trade_step
                self._last_recorded_trade_step = last_trade_step # Update our tracker
                trade_occurred_recently = True
        # --- End Fix check ---

        if not trade_occurred_recently:
             # Only increment if no NEW trade info was processed this step
             # Avoids double counting if step 0 has no trade info yet
             if self.current_step > 0 or self._last_recorded_trade_step != -1:
                  self.steps_since_last_market_trade += 1

        # --- Check conditions for switching to Skeleton mode ---
        if self.mode == 'background':
            # Ensure total_steps_in_period is available
            if not hasattr(self, 'total_steps_in_period') or self.total_steps_in_period is None:
                 self.logger.warning(f"{step_id}: total_steps_in_period not set. Cannot check mode switch.")
                 return # Cannot proceed without total steps info

            remaining_steps = self.total_steps_in_period - self.current_step - 1
            if remaining_steps < 0: remaining_steps = 0

            # Calculate thresholds for switching
            time_threshold = self.switch_time_threshold
            fraction_threshold = self.switch_remain_frac * remaining_steps
            # Ensure threshold is at least 0
            switch_point = max(0, min(time_threshold, fraction_threshold))

            if self.steps_since_last_market_trade > switch_point:
                self.logger.debug(f"{step_id}: RG Switching to Skeleton mode. Inactivity={self.steps_since_last_market_trade} > threshold={switch_point:.2f}")
                self.mode = 'skeleton'
            # else: remain in background
    # --- END CORRECTED _check_and_update_mode ---


    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Decide action based on current mode (Background/Skeleton). """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        self._check_and_update_mode(market_history) # Update timers and mode first

        if not self.can_trade(): return None
        if self.acted_this_period: return None # Don't quote if already acted

        # Get the value for the NEXT potential trade (uses BaseTrader logic)
        value = self.get_next_value_cost()
        if value is None: return None # No more tokens or invalid state

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        final_bid = None # Initialize action to None

        if self.mode == 'background':
            # --- Background Mode: INTERPRETED "Random Overbid" Snipe ---
            # WARNING: This overbidding logic is an interpretation and may need refinement.
            if current_ask is not None:
                try:
                    ask_f = float(current_ask)
                    # Target price: slightly above the ask, capped by own value
                    target_bid = min(value, ask_f + self.overbid_amount)
                    target_bid = int(round(target_bid))
                    # Ensure profitable and within bounds
                    bid_price = max(self.min_price, min(self.max_price, target_bid))
                    bid_price = min(bid_price, value) # Final profit check vs own value

                    # Check if this bid improves the current market bid
                    current_bid_f = float(current_bid) if current_bid is not None else -np.inf # Use -inf if no bid
                    is_improving = (bid_price > current_bid_f)

                    if bid_price >= self.min_price and is_improving:
                        final_bid = bid_price # Set the final action
                        self.logger.debug(f"{step_id}: RG (Background) proposing overbid snipe {final_bid} (Ask={ask_f}, Value={value}). Setting acted=True.")
                        self.acted_this_period = True # Set flag only if submitting
                    # else: # Optional debugging for why snipe failed
                    #    if not (bid_price >= self.min_price): self.logger.debug(f"{step_id} BG Snipe fail: Bid < min_price")
                    #    if not is_improving: self.logger.debug(f"{step_id} BG Snipe fail: Doesn't improve {current_bid}")

                except (ValueError, TypeError) as e:
                     self.logger.warning(f"{step_id}: RG (Background) error during snipe calc: {e}")
            # else: No ask to snipe

        elif self.mode == 'skeleton':
            # --- Skeleton Mode: Use the assumed SK logic ---
            self.logger.debug(f"{step_id}: RG acting in Skeleton mode.")
            if self._sk_logic is None:
                 self.logger.error("RG Skeleton mode entered but _sk_logic is None!")
                 return None

            # --- FIX: Set up helper state correctly ---
            # Store original state of helper
            original_vals = self._sk_logic.private_values
            original_tokens_left = self._sk_logic.tokens_left
            original_max_tokens = self._sk_logic.max_tokens

            # Set helper state to reflect the *current* situation for the *next* trade
            # Ensure we access the correct index based on how BaseTrader stores values
            current_token_index = self.max_tokens - self.tokens_left
            if 0 <= current_token_index < len(self.private_values):
                 current_value = self.private_values[current_token_index]
                 self._sk_logic.private_values = [current_value] # List with only the current value
                 self._sk_logic.max_tokens = 1                   # It only knows about this one token now
                 self._sk_logic.tokens_left = 1                  # It has this one token left to trade
                 self._sk_logic.current_step = self.current_step # Sync step

                 # Call the helper's logic
                 sk_bid = self._sk_logic.make_bid_or_ask(current_bid_info, current_ask_info, phibid, phiask, market_history)

                 if sk_bid is not None:
                     final_bid = sk_bid # Set the final action
                     self.logger.debug(f"{step_id}: RG (Skeleton) proposing bid {final_bid}. Setting acted=True.")
                     self.acted_this_period = True # Set flag only if submitting
            else:
                 self.logger.error(f"{step_id}: RG Invalid index {current_token_index} accessing private_values (len={len(self.private_values)}). Cannot use Skeleton.")
                 sk_bid = None # Ensure sk_bid is None if index was bad
            # --- END FIX for helper state ---


            # --- FIX: Restore original helper state ---
            self._sk_logic.private_values = original_vals
            self._sk_logic.tokens_left = original_tokens_left
            self._sk_logic.max_tokens = original_max_tokens
            # --- END FIX ---


        return final_bid # Return the final action (bid or None)


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs value, only if haven't acted. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        if not self.can_trade() or self.acted_this_period: return False
        if current_offer_price is None: return False

        value = self.get_next_value_cost()
        if value is None: return False

        try: is_profitable = (float(current_offer_price) <= value)
        except (ValueError, TypeError): return False

        if is_profitable:
            self.logger.debug(f"{step_id}: RG accepting BUY at {current_offer_price} (Value={value}). Setting acted=True.")
            self.acted_this_period = True # Mark as acted if accepting
            self._clear_rl_step_state()
        return is_profitable

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False # Buyers don't accept sell requests


# ==============================================================================
# RGSeller Class - applying the same fixes
# ==============================================================================

class RGSeller(BaseTrader):
    """
    Ringuette Seller Strategy (Based on RPM paper description):
    - Waits in background initially.
    - Snipes by "randomly undercutting" bid (interpretation needed).
    - Switches to Skeleton mode based on inactivity timer.
    - Accepts profitable trades.
    - Revised: Only sets 'acted' flag if a quote is submitted or a trade accepted.
    - Revised: Fixed KeyError and IndexError in helper logic.
    """
    def __init__(self, name, is_buyer, private_values, **kwargs):
        super().__init__(name, False, private_values, strategy="rg")
        self.logger = logging.getLogger(f'trader.{self.name}')

        # Parameters for switching modes
        self.switch_time_threshold = kwargs.get('rg_switch_time', 12)
        self.switch_remain_frac = kwargs.get('rg_switch_frac', 0.6)
        # Parameters for skeleton mode
        self.sk_margin = kwargs.get('rg_sk_margin', 0.05)
        self.sk_shout_prob = kwargs.get('rg_sk_shout_prob', 0.9)
        # Parameter for undercutting (interpretation - needs tuning)
        self.undercut_amount = kwargs.get('rg_undercut', 1) # Simple -1 undercut

        # Internal State
        self.mode = 'background'
        self.steps_since_last_market_trade = 0
        self.acted_this_period = False
        self._last_recorded_trade_step = -1

        # Instantiate skeleton logic helper
        try:
            # Ensure SkeletonSellerLogic is the correct class name from sk.py
            self._sk_logic = SkeletonSellerLogic(f"{name}_sk_helper", False, [0],
                                              fixed_margin=self.sk_margin,
                                              shout_probability=self.sk_shout_prob)
            self._sk_logic.update_market_params(self.min_price, self.max_price)
        except NameError:
            self._sk_logic = None
            self.logger.error("RG Seller could not instantiate SkeletonLogic helper.")

        self.logger.debug(f"Initialized RG Seller: SwitchT={self.switch_time_threshold}, SwitchFrac={self.switch_remain_frac}, Undercut={self.undercut_amount}")

    def reset_for_new_period(self, round_idx, period_idx):
        """ Reset period-specific state. """
        super().reset_for_new_period(round_idx, period_idx)
        self.mode = 'background'
        self.steps_since_last_market_trade = 0
        self.acted_this_period = False
        self._last_recorded_trade_step = -1
        if self._sk_logic:
            self._sk_logic.total_steps_in_period = total_steps_in_period
            self._sk_logic.current_period = period_idx
            self._sk_logic.current_round = round_idx
        # self.logger.debug(f"RG Seller {self.name} reset for P{period_idx}. Acted={self.acted_this_period}, Mode={self.mode}")


    def record_trade(self, period, step, price):
        profit = super().record_trade(period, step, price)
        # Note: We don't reset market timer here based on own trade
        return profit

    # --- CORRECTED _check_and_update_mode ---
    # (Identical logic to RGBuyer's corrected version)
    def _check_and_update_mode(self, market_history):
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        last_trade_info = market_history.get('last_trade_info_for_period')

        trade_occurred_recently = False
        if last_trade_info and isinstance(last_trade_info, dict) and 'step' in last_trade_info:
            last_trade_step = last_trade_info['step']
            if last_trade_step > self._last_recorded_trade_step:
                self.steps_since_last_market_trade = self.current_step - last_trade_step
                self._last_recorded_trade_step = last_trade_step
                trade_occurred_recently = True

        if not trade_occurred_recently:
             if self.current_step > 0 or self._last_recorded_trade_step != -1:
                  self.steps_since_last_market_trade += 1

        if self.mode == 'background':
            if not hasattr(self, 'total_steps_in_period') or self.total_steps_in_period is None:
                 self.logger.warning(f"{step_id}: total_steps_in_period not set. Cannot check mode switch.")
                 return

            remaining_steps = self.total_steps_in_period - self.current_step - 1
            if remaining_steps < 0: remaining_steps = 0
            time_threshold = self.switch_time_threshold
            fraction_threshold = self.switch_remain_frac * remaining_steps
            switch_point = max(0, min(time_threshold, fraction_threshold))

            if self.steps_since_last_market_trade > switch_point:
                self.logger.debug(f"{step_id}: RG Switching to Skeleton mode. Inactivity={self.steps_since_last_market_trade} > threshold={switch_point:.2f}")
                self.mode = 'skeleton'
    # --- END CORRECTED _check_and_update_mode ---

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Decide action based on current mode (Background/Skeleton). """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        self._check_and_update_mode(market_history)

        if not self.can_trade() or self.acted_this_period: return None

        cost = self.get_next_value_cost()
        if cost is None: return None

        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None

        final_ask = None # Initialize action

        if self.mode == 'background':
            # --- Background Mode: INTERPRETED "Random Undercut" Snipe ---
            if current_bid is not None:
                try:
                    bid_f = float(current_bid)
                    # Target price: slightly below the bid, but >= own cost
                    target_ask = max(cost, bid_f - self.undercut_amount)
                    target_ask = int(round(target_ask))
                    # Ensure profitable and within bounds
                    ask_price = max(self.min_price, min(self.max_price, target_ask))
                    ask_price = max(ask_price, cost) # Final profit check vs cost

                    # Check if this ask improves the current market ask
                    current_ask_f = float(current_ask) if current_ask is not None else np.inf # Use inf if no ask
                    is_improving = (ask_price < current_ask_f)

                    if ask_price <= self.max_price and is_improving:
                        final_ask = ask_price # Set final action
                        self.logger.debug(f"{step_id}: RG (Background) proposing undercut snipe {final_ask} (Bid={bid_f}, Cost={cost}). Setting acted=True.")
                        self.acted_this_period = True # Set flag only if submitting
                    # else: # Debugging fails
                    #     if not (ask_price <= self.max_price): self.logger.debug(f"{step_id} BG Snipe fail: Ask > max_price")
                    #     if not is_improving: self.logger.debug(f"{step_id} BG Snipe fail: Doesn't improve {current_ask}")

                except (ValueError, TypeError) as e:
                     self.logger.warning(f"{step_id}: RG (Background) error during snipe calc: {e}")
            # else: No bid to undercut

        elif self.mode == 'skeleton':
            # --- Skeleton Mode: Use the assumed SK logic ---
            self.logger.debug(f"{step_id}: RG acting in Skeleton mode.")
            if self._sk_logic is None:
                 self.logger.error("RG Skeleton mode entered but _sk_logic is None!")
                 return None

            # --- FIX: Set up helper state correctly ---
            original_vals = self._sk_logic.private_values
            original_tokens_left = self._sk_logic.tokens_left
            original_max_tokens = self._sk_logic.max_tokens

            current_token_index = self.max_tokens - self.tokens_left
            if 0 <= current_token_index < len(self.private_values):
                 current_cost = self.private_values[current_token_index] # Get the correct cost
                 self._sk_logic.private_values = [current_cost] # List with only the current cost
                 self._sk_logic.max_tokens = 1                  # It only knows about this one token now
                 self._sk_logic.tokens_left = 1                 # It has this one token left to trade
                 self._sk_logic.current_step = self.current_step # Sync step

                 # Call the helper's logic
                 sk_ask = self._sk_logic.make_bid_or_ask(current_bid_info, current_ask_info, phibid, phiask, market_history)

                 if sk_ask is not None:
                     final_ask = sk_ask # Set the final action
                     self.logger.debug(f"{step_id}: RG (Skeleton) proposing ask {final_ask}. Setting acted=True.")
                     self.acted_this_period = True # Set flag only if submitting
            else:
                 self.logger.error(f"{step_id}: RG Invalid index {current_token_index} accessing private_values (len={len(self.private_values)}). Cannot use Skeleton.")
                 sk_ask = None
            # --- END FIX for helper state ---

            # --- FIX: Restore original helper state ---
            self._sk_logic.private_values = original_vals
            self._sk_logic.tokens_left = original_tokens_left
            self._sk_logic.max_tokens = original_max_tokens
            # --- END FIX ---

        return final_ask


    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Accept if profitable vs cost, only if haven't acted. """
        step_id = f"R{self.current_round}P{self.current_period}S{self.current_step}"
        if not self.can_trade() or self.acted_this_period: return False
        if current_bid_price is None: return False

        cost = self.get_next_value_cost()
        if cost is None: return False

        try: is_profitable = (float(current_bid_price) >= cost)
        except (ValueError, TypeError): return False

        if is_profitable:
            self.logger.debug(f"{step_id}: RG accepting SELL at {current_bid_price} (Cost={cost}). Setting acted=True.")
            self.acted_this_period = True # Set flag only if accepting
            self._clear_rl_step_state()
        return is_profitable
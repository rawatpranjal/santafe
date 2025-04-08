# traders/base.py
import logging
import random # Only needed if strategies use it directly
import numpy as np # Added for potential type hinting / checks

class BaseTrader:
    """
    Base class for all trading agents in the SFI Double Auction.
    Holds participant state and defines the interface for strategy decisions.
    """
    def __init__(self, name, is_buyer, private_values, strategy="base"):
        self.name = name # Keep original name (e.g., "B0", "S1")
        try:
            # Attempt to create a unique numeric ID, fallback to hash
            numeric_part = name[1:]
            self.id_numeric = int(numeric_part) if len(name) > 1 and numeric_part.isdigit() else hash(name) % (10**8) # Modulo hash
        except ValueError:
            self.id_numeric = hash(name) % (10**8) # Fallback if name isn't standard B#/S#

        self.strategy = strategy
        self.is_buyer = is_buyer

        # Ensure private values are integers
        if not all(isinstance(v, (int, float, np.number)) for v in private_values):
             logging.error(f"Trader {self.name}: Received non-numeric private values: {private_values}")
             pv_int = []
        elif not all(isinstance(v, int) for v in private_values):
             logging.warning(f"Trader {self.name}: Converting non-integer private values {private_values} to int.")
             pv_int = [int(round(v)) for v in private_values]
        else:
            pv_int = list(private_values) # Make a copy

        # Sort values/costs appropriately
        if is_buyer:
            self.private_values = sorted(pv_int, reverse=True)
        else:
            self.private_values = sorted(pv_int)

        self.max_tokens = len(private_values) # N
        self.tokens_left = 0 # Initialized in reset_for_new_period

        self.current_round_profit = 0.0 # Accumulates within a round
        self.total_game_profit = 0.0 # Accumulates across rounds

        self.current_round = -1
        self.current_period = -1
        self.current_step = -1
        self.current_substep = ""
        self.total_steps_in_period = 0
        self.min_price = 1      # Default, updated by update_market_params
        self.max_price = 1000   # Default, updated by update_market_params

        # Internal state for RL agents to track last action for reward assignment
        self._last_state = None
        self._last_action_idx = None

        logging.debug(f"Initialized Trader: {self.name}, Role: {'Buyer' if is_buyer else 'Seller'}, Strategy: {self.strategy}, Values/Costs: {self.private_values}")

    def update_market_params(self, min_price, max_price):
        """Update market price bounds."""
        self.min_price = int(min_price)
        self.max_price = int(max_price)

    def reset_for_new_round(self):
        """Reset profit accumulator for a new round."""
        logging.debug(f"Trader {self.name}: Resetting for new round.")
        self.current_round_profit = 0.0
        self._last_state = None # Clear RL state tracking at round start
        self._last_action_idx = None

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """Reset tokens and timing info for a new period within a round."""
        logging.debug(f"Trader {self.name}: Resetting for new period (R{round_idx} P{period_idx}).")
        self.tokens_left = self.max_tokens
        self.total_steps_in_period = total_steps_in_period
        self.current_round = round_idx
        self.current_period = period_idx
        self.current_step = -1 # Step counter updated by Auction
        self.current_substep = "" # Substep updated by Auction

    def can_trade(self):
        """Check if the trader has tokens remaining."""
        return self.tokens_left > 0

    def get_next_value_cost(self):
        """Get the value/cost for the next available token."""
        if not self.can_trade():
            return None
        # Index based on tokens *used*
        idx = self.max_tokens - self.tokens_left
        if 0 <= idx < len(self.private_values):
            return self.private_values[idx]
        else:
            logging.error(f"Trader {self.name}: Invalid index {idx} accessing private_values (len={len(self.private_values)}, tokens_left={self.tokens_left}).")
            return None

    def record_trade(self, period, step, price):
        """Record a successful trade, update profit and tokens."""
        logger = logging.getLogger(f'trader.{self.name}')
        if not self.can_trade():
            logger.error(f"Trade recorded at P{period} S{step} but no tokens left!")
            return 0.0

        val_cost = self.get_next_value_cost()
        if val_cost is None:
            logger.error(f"Trade recorded at P{period} S{step} but no value/cost found!")
            return 0.0

        # Ensure price is integer for profit calculation
        if not isinstance(price, int):
            logger.warning(f"Received non-integer price {price} for trade. Rounding.")
            price = int(round(price))

        # Calculate profit increment for this trade
        profit_increment = (val_cost - price) if self.is_buyer else (price - val_cost)

        # Update state
        self.tokens_left -= 1
        self.current_round_profit += profit_increment
        self.total_game_profit += profit_increment

        # <<< Log changed to DEBUG >>>
        logger.debug(f"TRADE Recorded @ P{period} S{step}: Price={price}. Val/Cost={val_cost}, ProfitInc={profit_increment:.2f}. TokensLeft={self.tokens_left}, RoundProfit={self.current_round_profit:.2f}")
        return profit_increment # Return the profit from this specific trade

    # --- Abstract Methods for Strategy Implementation (Updated Signatures) ---
    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Strategy determines a bid (buyer) or ask (seller) price to submit.
        Return price (int) or None to not submit.
        RL agents should store _last_state and _last_action_idx here.
        """
        raise NotImplementedError(f"'make_bid_or_ask' not implemented for {self.strategy}")

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Strategy decides whether to accept the current standing offer price (ask).
        Return True to accept, False otherwise.
        RL agents should store _last_state and _last_action_idx here.
        """
        raise NotImplementedError(f"'request_buy' not implemented for {self.strategy}")

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Strategy decides whether to accept the current standing bid price.
        Return True to accept, False otherwise.
        RL agents should store _last_state and _last_action_idx here.
        """
        raise NotImplementedError(f"'request_sell' not implemented for {self.strategy}")

    # --- Optional Methods for RL ---
    def set_mode(self, training=True):
        """Optional: Set agent mode (training/evaluation)."""
        pass # Base class does nothing

    def observe_reward(self, last_state, action_idx, reward, next_state, done):
        """Optional: Process observation for RL agents. Called by Auction."""
        pass # Base class does nothing

    def get_last_episode_stats(self):
        """Optional: Get stats from the last completed episode. Called by Auction for monitoring."""
        return {} # Base class returns empty dict
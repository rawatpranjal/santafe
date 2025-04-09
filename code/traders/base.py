# traders/base.py
import logging
import random
import numpy as np

class BaseTrader:
    """
    Base class for all trading agents in the SFI Double Auction.
    Modified to support Round=Episode structure.
    """
    def __init__(self, name, is_buyer, private_values, strategy="base"):
        self.name = name
        try: # Create a numeric ID for seeding, fallback to hash
            numeric_part = name[1:]
            self.id_numeric = int(numeric_part) if len(name) > 1 and numeric_part.isdigit() else hash(name) % (10**8)
        except ValueError:
            self.id_numeric = hash(name) % (10**8)

        self.strategy = strategy
        self.is_buyer = is_buyer
        self.logger = logging.getLogger(f'trader.{self.name}') # Logger for each agent instance

        # Store original values for resets (important for EL)
        self._original_private_values = tuple(private_values) # Use tuple for immutability
        self.private_values = list(private_values) # Working copy

        # Ensure private values are integers and sorted correctly
        if not isinstance(self.private_values, (list, np.ndarray)) or not all(isinstance(v, (int, float, np.number)) for v in self.private_values):
             self.logger.error(f"Invalid private values: {private_values}")
             self.private_values = []
             self._original_private_values = tuple()
        elif not all(isinstance(v, int) for v in self.private_values):
             self.private_values = [int(round(v)) for v in self.private_values]
             self._original_private_values = tuple(self.private_values) # Store int version
        else:
             self.private_values = list(self.private_values) # Ensure it's a list copy

        if is_buyer: self.private_values.sort(reverse=True)
        else: self.private_values.sort()
        self._original_private_values = tuple(self.private_values) # Store sorted tuple

        self.max_tokens = len(self.private_values)
        self.tokens_left = 0 # Initialized in reset_for_new_period

        # Profit tracking
        self.current_period_profit = 0.0 # Profit within the current period
        self.current_round_profit = 0.0  # Accumulates profit within a round (episode)
        self.total_game_profit = 0.0   # Accumulates across all rounds

        # Time/State Tracking
        self.current_round = -1
        self.current_period = -1
        self.current_step = -1
        self.current_substep = ""
        self.total_steps_in_period = 0
        self.min_price = 1
        self.max_price = 1000

        # Internal state for RL agents (cleared appropriately)
        self._current_step_state = None
        self._current_step_action = None
        self._current_step_log_prob = None
        self._current_step_value = None

        # self.logger.debug(f"Initialized BaseTrader {self.name}")

    def update_market_params(self, min_price, max_price):
        self.min_price = int(min_price)
        self.max_price = int(max_price)

    def reset_for_new_round(self):
        """ FULL Reset for a new round (new market conditions / RL episode). """
        # self.logger.debug(f"FULL reset for new round {self.current_round + 1}.")
        # Accumulate total profit before resetting round profit
        self.total_game_profit += self.current_round_profit
        self.current_round_profit = 0.0
        self.current_period_profit = 0.0 # Reset period profit too
        # Reset strategy-specific learning state (implemented by subclasses)
        self._reset_learning_state()
        # Clear temporary RL step state
        self._clear_rl_step_state()

    def reset_for_new_period(self, total_steps_in_period, round_idx, period_idx):
        """ Reset only PERIOD-specific state. Keep learned state. """
        # self.logger.debug(f"Resetting for period {period_idx} in round {round_idx}.")
        self.tokens_left = self.max_tokens # Reset tokens available for the period
        self.current_period_profit = 0.0 # Reset period profit tracker
        self.total_steps_in_period = total_steps_in_period
        self.current_round = round_idx
        self.current_period = period_idx
        self.current_step = -1 # Reset step counter for the period
        self.current_substep = ""
        # DO NOT reset learned parameters or persistent state (like LSTM hidden state) here
        self._clear_rl_step_state() # Clear temporary step state only

    def _reset_learning_state(self):
        """ Abstract method for strategy-specific reset of learned parameters/state. """
        # Called by reset_for_new_round()
        pass # Base class does nothing, subclasses override if they learn

    def _clear_rl_step_state(self):
         """ Helper to clear internal RL state related to a single step's action choice. """
         self._current_step_state = None
         self._current_step_action = None
         self._current_step_log_prob = None
         self._current_step_value = None

    def can_trade(self):
        """ Check if the trader has tokens remaining for the current period. """
        return self.tokens_left > 0

    def get_next_value_cost(self):
        """ Get the value/cost for the next available token IN THIS PERIOD. """
        if not self.can_trade():
            return None
        # Index based on tokens *used* in this period
        idx = self.max_tokens - self.tokens_left
        if 0 <= idx < len(self.private_values):
            # Return the original value/cost for this token index
            # EL strategy will use its internal reservation_prices instead
            return self.private_values[idx]
        else:
            self.logger.error(f"Invalid index {idx} accessing private_values (len={len(self.private_values)}, tokens_left={self.tokens_left}).")
            return None

    def record_trade(self, period, step, price):
        """ Records a trade, updates profits & token count. Returns profit_increment or None. """
        if not self.can_trade():
            self.logger.error(f"Trade recorded at P{period} S{step} but no tokens left!")
            return None # Indicate failure

        # Use the specific reservation price logic for EL, otherwise use base value/cost
        if hasattr(self, '_get_current_reservation_price'):
             # EL uses its current reservation price for profit calculation basis
             val_cost_basis = self._get_current_reservation_price()
             if val_cost_basis is None: # Should only happen if EL runs out of res prices
                  self.logger.error(f"Could not get reservation price basis for trade P{period} S{step}")
                  val_cost_basis = self.get_next_value_cost() # Fallback? Or fail? Let's use base value.
                  if val_cost_basis is None: return None # Fail if base value also missing
        else:
             # Non-EL agents use their fundamental value/cost
             val_cost_basis = self.get_next_value_cost()
             if val_cost_basis is None:
                  self.logger.error(f"Trade recorded at P{period} S{step} but no value/cost found!")
                  return None

        # Ensure price is valid numeric type
        try: price = int(round(float(price)))
        except (ValueError, TypeError):
             self.logger.warning(f"Received non-numeric price {price} for trade. Cannot record.")
             return None

        # Calculate profit increment based on the determined basis (value/cost or reservation)
        profit_increment = (val_cost_basis - price) if self.is_buyer else (price - val_cost_basis)

        # Update state *after* getting necessary info
        token_index_traded = self.max_tokens - self.tokens_left # Index of the token being traded
        self.tokens_left -= 1
        self.current_period_profit += profit_increment
        self.current_round_profit += profit_increment
        # self.total_game_profit gets updated in reset_for_new_round

        # Mark success for EL agent if applicable (implementation in EL class)
        if hasattr(self, 'trade_success') and 0 <= token_index_traded < len(self.trade_success):
            self.trade_success[token_index_traded] = True

        # self.logger.debug(f"TRADE Recorded @ P{period} S{step}: Price={price}. Basis={val_cost_basis}, Inc={profit_increment:.2f}. TokensLeft={self.tokens_left}, PeriodProfit={self.current_period_profit:.2f}, RoundProfit={self.current_round_profit:.2f}")
        return profit_increment


    # --- Abstract & Optional Methods ---
    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        raise NotImplementedError(f"'make_bid_or_ask' not implemented for {self.strategy}")

    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        raise NotImplementedError(f"'request_buy' not implemented for {self.strategy}")

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        raise NotImplementedError(f"'request_sell' not implemented for {self.strategy}")

    # Optional methods used by specific strategies / RL agents
    def set_mode(self, training=True): pass # For RL agents
    def observe_reward(self, last_state, action_idx, reward, next_state, done, step_outcome=None): pass # For RL agents
    def get_last_episode_stats(self): return {} # For RL agents
    def update_end_of_period(self, period_stats): pass # For EL agent
    def save_model(self, path_prefix): self.logger.warning(f"Save model not implemented for strategy {self.strategy}")
    def load_model(self, path_prefix): self.logger.warning(f"Load model not implemented for strategy {self.strategy}")
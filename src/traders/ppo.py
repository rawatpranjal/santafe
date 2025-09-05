# traders/ppo.py
import random
import logging
import numpy as np
import torch
import os
from collections import deque

from .base import BaseTrader
# Import the new core components
from .ppo_lstm_core import LSTMAgent, PPOLogicLSTM # Using LSTM core

class PPOTrader(BaseTrader): # Consider renaming to PPO_LSTM_Trader if preferred
    """ Interface layer connecting PPOLogicLSTM to the auction environment """
    def __init__(self, name, is_buyer, private_values, rl_config, **kwargs):
        strategy_name = "ppo_lstm" # Updated strategy name
        super().__init__(name, is_buyer, private_values, strategy=strategy_name)
        self.logger = logging.getLogger(f'trader.{self.name}')
        # Set logger level based on config (avoids flooding console if RL log level is high)
        rl_log_level_str = rl_config.get('log_level_rl', 'WARNING').upper()
        rl_log_level = getattr(logging, rl_log_level_str, logging.WARNING)
        self.logger.setLevel(rl_log_level) # Set level for this specific agent's logger

        self.rl_config = rl_config
        self.episode_rewards_raw = [] # Track rewards within the current episode (period)

        # --- State and Action Space ---
        # Determine information level from config
        self.information_level = rl_config.get("information_level", "base")
        # Calculate state dimension based on info level
        self.state_dim = self._calculate_state_dim(self.information_level, rl_config)
        self.logger.info(f"Trader {self.name} using info level '{self.information_level}' with state_dim={self.state_dim}")

        # Action space parameters
        self.num_price_actions = kwargs.get('num_price_actions', 21)
        if self.num_price_actions <= 0:
            self.logger.warning(f"num_price_actions ({self.num_price_actions}) must be positive. Setting to 1.")
            self.num_price_actions = 1
        self.action_dim = self.num_price_actions + 1 # +1 for "pass" action
        self.price_range_pct = kwargs.get('price_range_pct', 0.15) # Pct of market range used for action mapping

        # --- RL Agent Logic (Using LSTM core) ---
        # Create a unique seed for this agent instance
        base_seed = rl_config.get("rng_seed_rl", random.randint(0, 1000000))
        agent_seed = base_seed + self.id_numeric # Use numeric ID for offset
        # Instantiate the LSTM logic class, passing necessary parameters
        self.logic = PPOLogicLSTM(self.state_dim, self.action_dim, rl_config, agent_seed)

        # Initialize LSTM state (batch_size=1 for inference during interaction)
        self.current_lstm_state = self.logic.agent.get_initial_state(batch_size=1, device=self.logic.device)
        self.initial_episode_lstm_state = None # Stores hidden state at start of episode for BPTT

        # --- Model Loading ---
        load_path_prefix = rl_config.get("load_rl_model_path")
        if load_path_prefix:
            # Attempt to load agent-specific model first, then shared model
            agent_specific_prefix = f"{load_path_prefix}_{self.name}"
            shared_prefix = load_path_prefix
            loaded = False
            if os.path.exists(f"{agent_specific_prefix}_agent.pth"):
                 self.logic.load_model(agent_specific_prefix)
                 loaded = True
            elif os.path.exists(f"{shared_prefix}_agent.pth"):
                 self.logger.warning(f"Agent specific model for {self.name} not found, loading shared model: {shared_prefix}")
                 self.logic.load_model(shared_prefix)
                 loaded = True

            if not loaded:
                 self.logger.warning(f"Load model path specified, but no model file found for prefix: {shared_prefix} or {agent_specific_prefix}. Starting from scratch.")


    def _calculate_state_dim(self, info_level, config):
        """ Calculates the state dimension based on the information level. """
        # Define base feature dimensions
        own_state_dim = 3 # tokens_left_norm, next_val_cost_norm, is_holding_best_quote_flag
        time_dim = 1      # time_remaining_norm

        # Get parameters from config safely
        params = config.get('rl_params', {})
        lob_depth = params.get("lob_depth_ppo", 3) # LOB depth for open_book

        # Calculate dimension based on level
        if info_level == "base":
            base_market_dim = 4 # best_bid, best_ask, phibid, phiask
            return own_state_dim + time_dim + base_market_dim
        elif info_level == "open_book":
            lob_features = lob_depth * 2 # N prices for bids, N for asks
            return own_state_dim + time_dim + lob_features
        elif info_level == "minimal":
            minimal_market_dim = 1 # last_trade_price
            return own_state_dim + time_dim + minimal_market_dim
        else:
            # Fallback to base if info_level is unknown
            self.logger.error(f"Unknown information_level in config: {info_level}. Defaulting to 'base' state dim.")
            self.information_level = "base" # Correct the internal state
            base_market_dim = 4
            return own_state_dim + time_dim + base_market_dim

    def _get_state(self, market_info: dict):
        """ Get state vector based on configured information level. """
        # Initialize state vector with a default value (e.g., -1 for missing/invalid)
        state = np.full(self.state_dim, -1.0, dtype=np.float32)
        price_range = max(self.max_price - self.min_price, 1) # Avoid division by zero

        # --- Normalization Helper Functions ---
        def norm_price(p):
            """ Normalize price to [-1, 1], return -1 if invalid/None. """
            if p is None or not isinstance(p, (int, float, np.number)) or not (self.min_price <= p <= self.max_price):
                return -1.0
            return ((p - self.min_price) / price_range) * 2.0 - 1.0

        def norm_fraction(val, max_val):
            """ Normalize value to [-1, 1] based on max_val. """
            if max_val is None or max_val <= 0: return -1.0
            # Clip ensures value is within [0, 1] before scaling
            return np.clip(val / max_val, 0.0, 1.0) * 2.0 - 1.0

        # --- Build State Vector ---
        idx = 0
        try:
            # Time features
            state[idx] = norm_fraction(market_info.get('total_steps', 0) - market_info.get('step', 0), market_info.get('total_steps', 0))
            idx += 1
            # Own State features
            state[idx] = norm_fraction(self.tokens_left, self.max_tokens)
            idx += 1
            val_cost = self.get_next_value_cost()
            state[idx] = norm_price(val_cost)
            idx += 1
            # Holding best quote flag
            current_bid_info = market_info.get('current_bid_info')
            current_ask_info = market_info.get('current_ask_info')
            is_holding_bid = current_bid_info and current_bid_info.get('agent') == self
            is_holding_ask = current_ask_info and current_ask_info.get('agent') == self
            state[idx] = 1.0 if (self.is_buyer and is_holding_bid) or (not self.is_buyer and is_holding_ask) else -1.0
            idx += 1

            # Information Level Specific Features
            if self.information_level == "base":
                if idx + 4 > self.state_dim: raise IndexError("State dim too small for base info")
                best_bid = current_bid_info['price'] if current_bid_info else None
                best_ask = current_ask_info['price'] if current_ask_info else None
                state[idx] = norm_price(best_bid)
                idx += 1
                state[idx] = norm_price(best_ask)
                idx += 1
                state[idx] = norm_price(market_info.get('phibid'))
                idx += 1
                state[idx] = norm_price(market_info.get('phiask'))
                idx += 1

            elif self.information_level == "open_book":
                lob_depth = self.rl_config.get("rl_params", {}).get("lob_depth_ppo", 3)
                if idx + (lob_depth * 2) > self.state_dim: raise IndexError("State dim too small for open_book info")
                
                # Ensure bids/asks are lists of tuples (name, price)
                all_bids = market_info.get('all_bids', []) if isinstance(market_info.get('all_bids'), list) else []
                all_asks = market_info.get('all_asks', []) if isinstance(market_info.get('all_asks'), list) else []

                for i in range(lob_depth):
                    # Check index and tuple structure before accessing price
                    state[idx] = norm_price(all_bids[i][1]) if i < len(all_bids) and isinstance(all_bids[i], tuple) and len(all_bids[i]) > 1 else -1.0
                    idx += 1
                for i in range(lob_depth):
                    state[idx] = norm_price(all_asks[i][1]) if i < len(all_asks) and isinstance(all_asks[i], tuple) and len(all_asks[i]) > 1 else -1.0
                    idx += 1

            elif self.information_level == "minimal":
                if idx + 1 > self.state_dim: raise IndexError("State dim too small for minimal info")
                last_trade_info = market_info.get('last_trade_info')
                last_trade_price = last_trade_info['price'] if last_trade_info and isinstance(last_trade_info, dict) else None
                state[idx] = norm_price(last_trade_price)
                idx += 1

            # Final dimension check
            if idx != self.state_dim:
                self.logger.warning(f"State final check mismatch! Built {idx}, expected {self.state_dim}. State: {state[:idx]}")
                # Padding is implicit as array was initialized with -1.0

        except (IndexError, KeyError, TypeError) as e:
            self.logger.error(f"Error building state vector at index {idx} (expected dim {self.state_dim}): {e}", exc_info=True)
            # Return the partially filled state (padded with -1)
            pass

        # Ensure no NaNs and clip final state
        if np.isnan(state).any():
             self.logger.error(f"NaN detected in state vector: {state}. Replacing with -1.")
             state = np.nan_to_num(state, nan=-1.0, posinf=1.0, neginf=-1.0) # Use -1 for NaN
        state = np.clip(state, -1.0, 1.0)
        return state


    def _map_action_to_price(self, action_idx):
        """ Maps action index directly to absolute market prices - full freedom. """
        if action_idx == self.action_dim - 1: # 'pass' action is the last index
            return None

        val_cost = self.get_next_value_cost()
        if val_cost is None:
            return None # Can't quote without a reference

        num_price_levels = self.action_dim - 1
        if num_price_levels <= 0: return None

        # FULL MARKET MAPPING: Let PPO learn ANY price in the market range
        # Actions map linearly across entire [min_price, max_price] range
        # PPO will learn which prices are profitable through rewards
        
        # Map action index to absolute price
        price_step = (self.max_price - self.min_price) / max(num_price_levels - 1, 1)
        target_price = self.min_price + action_idx * price_step
        
        # Only apply profitability constraint (no trading at a loss)
        if self.is_buyer:
            # Can bid up to value (but PPO will learn to bid lower for profit)
            final_price = min(val_cost, int(round(target_price)))
        else:  # Seller  
            # Can ask down to cost (but PPO will learn to ask higher for profit)
            final_price = max(val_cost, int(round(target_price)))
            
        # Ensure within market bounds
        final_price = max(self.min_price, min(self.max_price, final_price))

        return final_price

    def reset_for_new_period(self, round_idx, period_idx):
        """ Resets agent state for a new trading period (episode). """
        super().reset_for_new_period(round_idx, period_idx)
        
        # Update learning schedules at the start of each round (first period only)
        if period_idx == 0 and hasattr(self.logic, 'update_schedule'):
            self.logic.update_schedule(round_idx)
        
        # Reset LSTM hidden state
        self.current_lstm_state = self.logic.agent.get_initial_state(batch_size=1, device=self.logic.device)
        # Store the initial state for this episode for use in BPTT during learning
        self.initial_episode_lstm_state = (self.current_lstm_state[0].clone().detach(), self.current_lstm_state[1].clone().detach())
        self.episode_rewards_raw = [] # Reset reward tracker

    def set_mode(self, training=True):
        """ Sets the agent's mode (training affects learning updates). """
        self.logic.set_mode(training)

    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ Agent determines its action (quote price or pass) for the current step. """
        if not self.can_trade():
             self._current_step_state = None; self._current_step_action = None
             self._current_step_log_prob = None; self._current_step_value = None
             return None

        # Construct the market information dictionary needed for state generation
        market_info = {
            'step': self.current_step,
            'total_steps': self.total_steps_in_period,
            'period': self.current_period,
            'total_periods': self.rl_config.get('num_periods', 1),
            'current_bid_info': current_bid_info,
            'current_ask_info': current_ask_info,
            'phibid': phibid,
            'phiask': phiask,
            'last_trade_info': market_history.get('last_trade_info_for_period'),
            'all_bids': market_history.get('all_bids_this_step', []), # Use get with default
            'all_asks': market_history.get('all_asks_this_step', []), # Use get with default
        }
        state = self._get_state(market_info)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.logic.device)

        # --- CORRECTED UNPACKING (expects 5 values now) ---
        action_idx_tensor, log_prob, entropy, value, new_lstm_state = self.logic.agent.get_action_value_and_state(
            state_tensor, self.current_lstm_state
        )
        # --- END CORRECTION ---
        action_idx = action_idx_tensor.item()

        # Store state info
        self._current_step_state = state
        self._current_step_action = action_idx
        self._current_step_log_prob = log_prob.item()
        self._current_step_value = value.item()
        # Entropy is calculated but not directly stored/used here, it's used in the PPO loss

        # Update the agent's LSTM state
        self.current_lstm_state = new_lstm_state

        final_price = self._map_action_to_price(action_idx)
        return final_price

    def observe_reward(self, last_state, action_idx, reward, next_state, done, step_outcome=None):
         """ Processes the outcome of the previous action and stores transition. Triggers learning if done. """
         reward = float(reward) # Ensure reward is float
         self.episode_rewards_raw.append(reward)

         # Check if the transition corresponds to the action chosen by the policy
         # Use the internally stored values from the make_bid_or_ask call
         if last_state is not None and action_idx is not None and action_idx == self._current_step_action:
             log_prob = self._current_step_log_prob
             value = self._current_step_value
             if log_prob is not None and value is not None:
                 # Store the transition in the PPO logic buffer
                 self.logic.store_transition(last_state, action_idx, reward, done, log_prob, value)
             else:
                 # This warning indicates an internal logic error (state/action mismatch)
                 self.logger.warning(f"ObserveReward: Missing log_prob/value for stored action {action_idx}. Skipping.")
         # else: If action_idx != self._current_step_action, it means the action wasn't from policy
             # (e.g., maybe an automatic accept action if that logic existed, or no action stored)
             # self.logger.debug(f"ObserveReward: Skipping transition store. Action {action_idx} != stored {self._current_step_action} or no stored state/action.")


         # Clear the temporary step state *after* it might have been used
         self._current_step_state = None
         self._current_step_action = None
         self._current_step_log_prob = None
         self._current_step_value = None

         # If the episode (period) is done, trigger the learning update
         if done:
             if self.logic.is_training and self.logic.trajectory_buffer: # Check if training and buffer has data
                 if self.initial_episode_lstm_state is None:
                      # This should not happen if reset_for_new_period worked correctly
                      self.logger.error("ObserveReward: Missing initial LSTM state for learning update!")
                 else:
                      # Provide the initial hidden state for the episode and the state AFTER the last step
                      last_step_next_state_np = next_state if next_state is not None else np.zeros(self.state_dim, dtype=np.float32)
                      self.logic.learn(self.initial_episode_lstm_state, last_step_next_state_np)

             # Log episode summary (optional)
             total_reward = sum(self.episode_rewards_raw)
             self.logger.debug(f"End Period R{self.current_round}P{self.current_period}. Total Raw Reward: {total_reward:.2f}. Steps: {len(self.episode_rewards_raw)}")

             # Reset episode-specific trackers (already done in reset_for_new_period)
             self.episode_rewards_raw = []
             self.initial_episode_lstm_state = None


    def get_last_episode_stats(self):
        """ Retrieves and clears the training statistics from the last learning update. """
        stats = self.logic.last_train_stats.copy() # Get stats from the logic module
        self.logic.last_train_stats = {} # Clear after retrieving
        # Return stats only if learning actually happened (indicated by presence of key metrics)
        if stats and 'avg_policy_loss' in stats and not np.isnan(stats['avg_policy_loss']):
             return stats
        else:
             return {} # Return empty dict if no learning occurred or stats are invalid

    # --- request_buy / request_sell ---
    # For PPO agents, the policy decides whether to trade (by quoting a price)
    # or not trade (by choosing the 'pass' action). These methods are not used
    # for acceptance decisions by the PPO agent itself.
    def request_buy(self, current_offer_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ PPO policy decides acceptance through actions, returning False here. """
        # self.logger.debug("PPO Buyer received request_buy call (ignored).")
        return False

    def request_sell(self, current_bid_price, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """ PPO policy decides acceptance through actions, returning False here. """
        # self.logger.debug("PPO Seller received request_sell call (ignored).")
        return False

    # --- Save/Load Model ---
    def save_model(self, path_prefix):
        """ Saves the agent's learned model state. """
        self.logic.save_model(path_prefix)

    def load_model(self, path_prefix):
        """ Loads the agent's learned model state. """
        self.logic.load_model(path_prefix)
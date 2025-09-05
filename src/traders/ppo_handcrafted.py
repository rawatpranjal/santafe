# traders/ppo_handcrafted.py
"""
PPOHandcraftedTrader: A feature-engineered PPO agent that uses carefully designed features
inspired by successful classical trading strategies (Kaplan, ZIP, GD, etc.).

This agent synthesizes key strategic signals from decades of trading research into a compact,
powerful state representation for reinforcement learning.
"""

import logging
import numpy as np
from collections import deque
from .ppo import PPOTrader
from .base import BaseTrader


class PPOHandcraftedTrader(PPOTrader):
    """
    PPO agent with handcrafted features inspired by classical trading strategies.
    Inherits from PPOTrader but overrides state generation to use expert-designed features.
    """
    
    def __init__(self, name, is_buyer, private_values, rl_config, **kwargs):
        # Override the information level to use handcrafted features
        rl_config = dict(rl_config)  # Make a copy to avoid modifying original
        rl_config['information_level'] = 'handcrafted'
        
        # Call parent constructor
        super().__init__(name, is_buyer, private_values, rl_config, **kwargs)
        
        # Update strategy name for clarity
        self.strategy = "ppo_handcrafted"
        
        # Override state dimension to match our handcrafted features
        self.state_dim = 12  # 12 carefully designed features
        
        # Reinitialize the PPO logic with correct state dimension
        # Use feedforward PPO (no LSTM) for handcrafted features
        import random
        from .ppo_core import PPOLogic
        base_seed = rl_config.get("rng_seed_rl", random.randint(0, 1000000))
        agent_seed = base_seed + self.id_numeric
        self.logic = PPOLogic(self.state_dim, self.action_dim, rl_config, agent_seed)
        
        # Initialize tracking variables for advanced features
        self.ewma_price = None  # Exponentially weighted moving average of trade prices
        self.ewma_alpha = kwargs.get('ewma_alpha', 0.3)  # Smoothing factor for EWMA
        self.last_shout_price = None  # Track our last submitted quote (ZIP-inspired)
        self.recent_ask_history = deque(maxlen=kwargs.get('gd_history_len', 20))  # For GD-style belief
        self.recent_bid_history = deque(maxlen=kwargs.get('gd_history_len', 20))  # For GD-style belief
        self.market_trade_history = deque(maxlen=10)  # For tracking market activity
        self.quote_count_history = deque(maxlen=5)  # For opponent fingerprinting
        self.steps_since_last_trade = 0  # For market inactivity tracking
        
        self.logger.info(f"Initialized PPOHandcraftedTrader {name} with 12 expert features")
    
    def reset_for_new_period(self, round_idx, period_idx):
        """Reset agent state for a new trading period."""
        # Call BaseTrader's reset_for_new_period with correct signature
        BaseTrader.reset_for_new_period(self, round_idx, period_idx)
        
        # Update learning schedules at the start of each round (first period only)
        if period_idx == 0 and hasattr(self.logic, 'update_schedule'):
            self.logic.update_schedule(round_idx)
        
        self.episode_rewards_raw = []
        
        # Reset tracking variables but keep some history for continuity within round
        self.ewma_price = None  # Reset EWMA at period start
        self.last_shout_price = None
        self.steps_since_last_trade = 0
        # Keep recent histories partially for within-round learning
        if period_idx == 0:  # New round - clear everything
            self.recent_ask_history.clear()
            self.recent_bid_history.clear()
            self.market_trade_history.clear()
            self.quote_count_history.clear()
    
    def _calculate_state_dim(self, info_level, config):
        """Override to return our fixed handcrafted state dimension."""
        return 12
    
    def _update_market_observations(self, market_info):
        """Update internal tracking based on current market state."""
        # Update bid/ask history for GD-style belief estimation
        current_bid_info = market_info.get('current_bid_info')
        current_ask_info = market_info.get('current_ask_info')
        
        if current_bid_info and isinstance(current_bid_info.get('price'), (int, float)):
            self.recent_bid_history.append(current_bid_info['price'])
        
        if current_ask_info and isinstance(current_ask_info.get('price'), (int, float)):
            self.recent_ask_history.append(current_ask_info['price'])
        
        # Update trade history and EWMA
        last_trade_info = market_info.get('last_trade_info_for_period')
        if last_trade_info and isinstance(last_trade_info, dict):
            trade_price = last_trade_info.get('price')
            if trade_price is not None:
                # Update EWMA
                if self.ewma_price is None:
                    self.ewma_price = float(trade_price)
                else:
                    self.ewma_price = self.ewma_alpha * float(trade_price) + (1 - self.ewma_alpha) * self.ewma_price
                
                # Update steps since trade
                last_trade_step = last_trade_info.get('step', -1)
                current_step = market_info.get('step', 0)
                self.steps_since_last_trade = current_step - last_trade_step if last_trade_step >= 0 else self.steps_since_last_trade + 1
        else:
            self.steps_since_last_trade += 1
        
        # Track quote activity for opponent fingerprinting
        all_bids = market_info.get('all_bids_this_step', [])
        all_asks = market_info.get('all_asks_this_step', [])
        total_quotes = len(all_bids) + len(all_asks)
        self.quote_count_history.append(total_quotes)
    
    def _estimate_acceptance_probability(self, is_buyer, potential_price):
        """
        GD-inspired: Estimate probability that a quote at potential_price would be accepted.
        """
        if is_buyer:
            # For buyer: probability that an ask <= potential_price will appear
            relevant_history = list(self.recent_ask_history)
            if not relevant_history:
                return 0.5  # No history, assume 50%
            accepting_count = sum(1 for ask in relevant_history if ask <= potential_price)
        else:
            # For seller: probability that a bid >= potential_price will appear
            relevant_history = list(self.recent_bid_history)
            if not relevant_history:
                return 0.5  # No history, assume 50%
            accepting_count = sum(1 for bid in relevant_history if bid >= potential_price)
        
        return accepting_count / len(relevant_history)
    
    def _get_state(self, market_info: dict):
        """
        Generate the handcrafted feature vector (12 dimensions).
        Features are inspired by successful classical trading strategies.
        """
        # Update observations first
        self._update_market_observations(market_info)
        
        # Initialize state vector
        state = np.full(self.state_dim, 0.0, dtype=np.float32)
        
        # Extract market information
        current_bid_info = market_info.get('current_bid_info')
        current_ask_info = market_info.get('current_ask_info')
        current_bid = current_bid_info['price'] if current_bid_info else None
        current_ask = current_ask_info['price'] if current_ask_info else None
        
        price_range = max(self.max_price - self.min_price, 1)
        total_steps = self.ntimes  # Use ntimes from game params
        current_step = self.current_step
        
        # Helper function for safe normalization
        def safe_normalize(value, min_val, max_val):
            if value is None or max_val <= min_val:
                return 0.0
            return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0) * 2.0 - 1.0
        
        # Get next value/cost
        val_cost = self.get_next_value_cost()
        
        idx = 0
        
        # ===== CORE FEATURES (4) =====
        
        # Feature 0: Time Remaining (normalized to [-1, 1])
        time_remaining_frac = (total_steps - current_step) / max(total_steps, 1)
        state[idx] = time_remaining_frac * 2.0 - 1.0
        idx += 1
        
        # Feature 1: Tokens Remaining (normalized to [-1, 1])
        tokens_remaining_frac = self.tokens_left / max(self.max_tokens, 1)
        state[idx] = tokens_remaining_frac * 2.0 - 1.0
        idx += 1
        
        # Feature 2: Am I Holding Best Quote? (binary: -1 or 1)
        is_holding_bid = current_bid_info and current_bid_info.get('agent') == self
        is_holding_ask = current_ask_info and current_ask_info.get('agent') == self
        state[idx] = 1.0 if (self.is_buyer and is_holding_bid) or (not self.is_buyer and is_holding_ask) else -1.0
        idx += 1
        
        # Feature 3: Inventory Risk (urgency indicator)
        # High value means too many tokens for time remaining
        if total_steps - current_step > 0:
            inventory_risk = (tokens_remaining_frac / time_remaining_frac) - 1.0
            state[idx] = np.clip(inventory_risk, -1.0, 1.0)
        else:
            state[idx] = 1.0 if self.tokens_left > 0 else -1.0
        idx += 1
        
        # ===== MARKET SIGNALS (5) =====
        
        # Feature 4: Bid-Ask Spread (Kaplan snipe signal)
        if current_bid is not None and current_ask is not None:
            spread = (current_ask - current_bid) / max(price_range, 1)
            state[idx] = np.clip(spread * 2.0 - 1.0, -1.0, 1.0)  # Normalize to [-1, 1]
        else:
            state[idx] = 1.0  # Max spread when one side missing
        idx += 1
        
        # Feature 5: Spread Dominance (where my value sits relative to market)
        if val_cost is not None and current_bid is not None and current_ask is not None:
            if self.is_buyer:
                if val_cost < current_bid:
                    dominance = -1.0  # Out of market (too low)
                elif val_cost > current_ask:
                    dominance = 1.0  # Can trade profitably now
                else:
                    # In the spread
                    dominance = (val_cost - current_bid) / max(current_ask - current_bid, 1)
            else:  # Seller
                if val_cost > current_ask:
                    dominance = -1.0  # Out of market (too high)
                elif val_cost < current_bid:
                    dominance = 1.0  # Can trade profitably now
                else:
                    # In the spread
                    dominance = (current_ask - val_cost) / max(current_ask - current_bid, 1)
            state[idx] = np.clip(dominance * 2.0 - 1.0, -1.0, 1.0)
        else:
            state[idx] = 0.0
        idx += 1
        
        # Feature 6: Price Trend (EWMA momentum, ZIP-inspired)
        if self.ewma_price is not None and val_cost is not None:
            # Positive means market trending above my value (for buyer, bad; for seller, good)
            trend = (self.ewma_price - val_cost) / max(price_range, 1)
            if self.is_buyer:
                trend = -trend  # Reverse for buyer perspective
            state[idx] = np.clip(trend * 4.0, -1.0, 1.0)  # Scale up for sensitivity
        else:
            state[idx] = 0.0
        idx += 1
        
        # Feature 7: Market Inactivity (Kaplan mode-switch signal)
        inactivity_normalized = min(self.steps_since_last_trade / max(total_steps * 0.1, 1), 1.0)
        state[idx] = inactivity_normalized * 2.0 - 1.0
        idx += 1
        
        # Feature 8: Empirical Acceptance Probability (GD-inspired belief)
        if val_cost is not None:
            # Use a default quote near our value to estimate acceptance
            if self.is_buyer:
                test_price = val_cost - price_range * 0.05  # Slightly below value
            else:
                test_price = val_cost + price_range * 0.05  # Slightly above cost
            accept_prob = self._estimate_acceptance_probability(self.is_buyer, test_price)
            state[idx] = accept_prob * 2.0 - 1.0
        else:
            state[idx] = 0.0
        idx += 1
        
        # ===== COMPETITIVE LANDSCAPE (3) =====
        
        # Feature 9: Average Quotes Per Step (opponent activity level)
        if len(self.quote_count_history) > 0:
            avg_quotes = np.mean(self.quote_count_history)
            # Normalize assuming typical range is 0-20 quotes per step
            state[idx] = np.clip((avg_quotes / 10.0) - 1.0, -1.0, 1.0)
        else:
            state[idx] = 0.0
        idx += 1
        
        # Feature 10: Quote Volatility (market type detection)
        if len(self.recent_bid_history) >= 3 and len(self.recent_ask_history) >= 3:
            bid_volatility = np.std(list(self.recent_bid_history)[-5:]) / max(price_range, 1)
            ask_volatility = np.std(list(self.recent_ask_history)[-5:]) / max(price_range, 1)
            avg_volatility = (bid_volatility + ask_volatility) / 2
            state[idx] = np.clip(avg_volatility * 10.0 - 1.0, -1.0, 1.0)  # Scale for sensitivity
        else:
            state[idx] = 0.0
        idx += 1
        
        # Feature 11: My Last Quote vs Last Trade (ZIP-style personal feedback)
        if self.last_shout_price is not None and self.ewma_price is not None:
            # For buyer: negative if my bid was too low, positive if too high
            # For seller: negative if my ask was too low, positive if too high
            quote_feedback = (self.last_shout_price - self.ewma_price) / max(price_range, 1)
            if self.is_buyer:
                quote_feedback = -quote_feedback  # Reverse for buyer
            state[idx] = np.clip(quote_feedback * 4.0, -1.0, 1.0)
        else:
            state[idx] = 0.0
        idx += 1
        
        # Ensure no NaNs or infinities
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -1.0, 1.0)
        
        return state
    
    def make_bid_or_ask(self, current_bid_info, current_ask_info, phibid, phiask, market_history):
        """
        Determine action using the handcrafted state representation.
        Also tracks the last shout price for ZIP-style feedback.
        """
        if not self.can_trade():
            self._current_step_state = None
            self._current_step_action = None
            self._current_step_log_prob = None
            self._current_step_value = None
            return None

        # Construct the market information dictionary needed for state generation
        market_info = {
            'step': self.current_step,
            'total_steps': self.ntimes,  # Use ntimes instead of total_steps_in_period
            'period': self.current_period,
            'total_periods': self.rl_config.get('num_periods', 1),
            'current_bid_info': current_bid_info,
            'current_ask_info': current_ask_info,
            'phibid': phibid,
            'phiask': phiask,
            'last_trade_info_for_period': market_history.get('last_trade_info_for_period'),
            'all_bids_this_step': market_history.get('all_bids_this_step', []),
            'all_asks_this_step': market_history.get('all_asks_this_step', []),
        }
        
        # Get state using our handcrafted features
        state = self._get_state(market_info)
        
        # Get action from feedforward PPO logic
        action_idx, log_prob, value = self.logic.get_action(state)
        
        # Store state info for learning
        self._current_step_state = state
        self._current_step_action = action_idx
        self._current_step_log_prob = log_prob
        self._current_step_value = value
        
        # Map action to price
        final_price = self._map_action_to_price(action_idx)
        
        # Track our last shout for ZIP-style feedback feature
        self.last_shout_price = final_price
        
        return final_price
    
    def observe_reward(self, last_state, action_idx, reward, next_state, done, step_outcome=None):
        """Store transition for learning with enhanced reward shaping."""
        if hasattr(self, '_current_step_state') and self._current_step_state is not None:
            # Apply reward shaping to encourage aggressive bidding
            shaped_reward = reward
            if reward > 0:  # Only shape positive rewards (successful trades)
                # Amplify larger profits more than smaller ones
                # This encourages the agent to seek higher profit margins
                max_possible_profit = (self.max_price - self.min_price) * 0.5
                profit_ratio = reward / max_possible_profit
                # Quadratic shaping: small profits stay small, large profits get amplified
                shaped_reward = reward * (1 + profit_ratio)
            
            self.logic.store_transition(
                self._current_step_state,
                self._current_step_action,
                shaped_reward,
                done,
                self._current_step_log_prob,
                self._current_step_value
            )
            self.episode_rewards_raw.append(reward)  # Track raw rewards for logging
    
    def update_end_of_period(self, period_trade_prices):
        """Called at end of period to trigger learning."""
        super().update_end_of_period(period_trade_prices)
        
        # Trigger learning at end of period if we're in training mode
        if hasattr(self.logic, 'is_training') and self.logic.is_training:
            # Get the next state for bootstrapping
            next_state = self._get_state({
                'step': self.current_step,
                'total_steps': self.ntimes,
                'period': self.current_period,
                'total_periods': 1,
                'current_bid_info': None,
                'current_ask_info': None,
                'phibid': 0,
                'phiask': 0,
                'last_trade_info_for_period': None,
                'all_bids_this_step': [],
                'all_asks_this_step': [],
            })
            
            # Perform PPO update
            self.logic.learn(next_state)
            
            # Log episode summary
            if self.episode_rewards_raw:
                total_reward = sum(self.episode_rewards_raw)
                self.logger.debug(f"End Period R{self.current_round}P{self.current_period}. "
                                f"Total Reward: {total_reward:.2f}. Steps: {len(self.episode_rewards_raw)}")
            
            self.episode_rewards_raw = []
    
    def set_mode(self, training=True):
        """Set training/evaluation mode."""
        if hasattr(self.logic, 'set_mode'):
            self.logic.set_mode(training)